########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

#!/bin/bash
set -e

BASEDIR=$(dirname "$0")
cd $BASEDIR
WORKINGDIR=`pwd`
source_dir=$(cd `dirname ${WORKINGDIR}`; pwd)

source ${WORKINGDIR}/.env

# fetch package info 
cd ${source_dir}
version=`grep "FATE=" .env | awk -F '=' '{print $2}'`
fateboard_version=`grep "FATEBOARD=" .env | awk -F '=' '{print $2}'`
package_dir_name="FATE_install_"${version}
package_dir=${source_dir}/cluster-deploy/${package_dir_name}
echo "[INFO] Build info"
echo "[INFO] version: "${version}
echo "[INFO] version tag: "${version_tag}
echo "[INFO] Package output dir is "${package_dir}
rm -rf ${package_dir} ${package_dir}-${version_tag}".tar.gz"
mkdir -p ${package_dir}

eggroll_git_url=`grep -A 3 '"eggroll"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
eggroll_git_branch=`grep -A 3 '"eggroll"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
fateboard_git_url=`grep -A 3 '"fateboard"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
fateboard_git_branch=`grep -A 3 '"fateboard"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`

package() {
  # create package path
  [ -d ${package_dir} ] && rm -rf ${package_dir}
  mkdir -p ${package_dir}/python/arch

  # package python
  echo "[INFO] Package fate start"
  cp -r arch/conf ${package_dir}/python/arch/
  cp -r arch/api ${package_dir}/python/arch/
  cp -r arch/transfer_variables ${package_dir}/python/arch/
  cp -r arch/standalone ${package_dir}/python/arch/
  cp .env requirements.txt RELEASE.md ${package_dir}/python/
  cp -r examples federatedml federatedrec fate_flow ${package_dir}/python/
  cp -r bin  ${package_dir}/
  echo "[INFO] Package fate done"
  echo "[INFO] Package fateboard start"

  cd ${source_dir}
  echo "[INFO] Git clone fateboard submodule source code from ${fateboard_git_url} branch ${fateboard_git_branch}"
  if [[ -e "fateboard" ]];then
      while [[ true ]];do
          read -p "The fateboard directory already exists, delete and re-download? [y/n] " input
          case ${input} in
          [yY]*)
                  echo "[INFO] Delete the original fateboard"
                  rm -rf fateboard
                  git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
                  break
                  ;;
          [nN]*)
                  echo "[INFO] Use the original fateboard"
                  break
                  ;;
          *)
                  echo "Just enter y or n, please."
                  ;;
          esac
      done
  else
      git clone ${fateboard_git_url} -b ${fateboard_git_branch} --depth=1 fateboard
  fi
  docker run --rm -u $(id -u):$(id -g) -v ${source_dir}/fateboard:/data/projects/fate/fateboard --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/fateboard && mvn clean package -DskipTests"
  cd ./fateboard
  mkdir -p ${package_dir}/fateboard/conf
  mkdir -p ${package_dir}/fateboard/ssh
  cp ./target/fateboard-${fateboard_version}.jar ${package_dir}/fateboard/
  cp ./bin/service.sh ${package_dir}/fateboard/
  cp ./src/main/resources/application.properties ${package_dir}/fateboard/conf/
  cd ${package_dir}/fateboard
  touch ./ssh/ssh.properties
  ln -s fateboard-${fateboard_version}.jar fateboard.jar
  echo "[INFO] Package fateboard done"
  
  echo "[INFO] Package fateboard start"
  cd ${source_dir}
  echo "[INFO] Git clone eggroll submodule source code from ${eggroll_git_url} branch ${eggroll_git_branch}"
  if [[ -e "eggroll" ]];then
      while [[ true ]];do
          read -p "The eggroll directory already exists, delete and re-download? [y/n] " input
          case ${input} in
          [yY]*)
                  echo "[INFO] Delete the original eggroll"
                  rm -rf eggroll
                  git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
                  break
                  ;;
          [nN]*)
                  echo "[INFO] Use the original eggroll"
                  break
                  ;;
          *)
                  echo "Just enter y or n, please."
                  ;;
          esac
      done
  else
      git clone ${eggroll_git_url} -b ${eggroll_git_branch} --depth=1 eggroll
  fi

  eggroll_source_code_dir=${source_dir}/eggroll
  docker run --rm -u $(id -u):$(id -g) -v ${eggroll_source_code_dir}:/data/projects/fate/eggroll --entrypoint="" maven:3.6-jdk-8 /bin/bash -c "cd /data/projects/fate/eggroll/deploy && bash auto-packaging.sh "
  mkdir -p ${package_dir}/eggroll
  mv ${source_dir}/eggroll/eggroll.tar.gz ${package_dir}/eggroll/
  cd ${package_dir}/eggroll/
  tar xzf eggroll.tar.gz
  rm -rf eggroll.tar.gz
  echo "[INFO] Package eggroll done"
}

buildBase() {
  [ -f ${source_dir}/docker-build/docker/base/requirements.txt ] && rm ${source_dir}/docker-build/docker/base/requirements.txt
  ln ${source_dir}/requirements.txt ${source_dir}/docker-build/docker/base/requirements.txt
  echo "START BUILDING BASE IMAGE"
  cd ${WORKINGDIR}

  docker build --build-arg version=${version} -f docker/base/Dockerfile -t ${PREFIX}/base-image:${BASE_TAG} ${source_dir}/docker-build/docker/base

  rm ${source_dir}/docker-build/docker/base/requirements.txt
  echo "FINISH BUILDING BASE IMAGE"
}

buildModule() {
  [ -f ${source_dir}/docker-build/docker/modules/federation/fate-federation-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/federation/fate-federation-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/proxy/fate-proxy-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/proxy/fate-proxy-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/roll/eggroll-roll-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/roll/eggroll-roll-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/meta-service/eggroll-meta-service-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/meta-service/eggroll-meta-service-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/fateboard/fateboard-*.jar ] && rm ${source_dir}/docker-build/docker/modules/fateboard/fateboard-*.jar
  [ -f ${source_dir}/docker-build/docker/modules/egg/eggroll-api-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/egg/eggroll-api-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/egg/eggroll-conf-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/egg/eggroll-conf-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/egg/eggroll-computing-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/egg/eggroll-computing-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/egg/eggroll-egg-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/egg/eggroll-egg-*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/egg/eggroll-storage-service-cxx-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/egg/eggroll-storage-service-cxx*.tar.gz
  [ -f ${source_dir}/docker-build/docker/modules/egg/third_party_eggrollv1.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/egg/third_party_eggrollv1.tar.gz
  [ -d ${source_dir}/docker-build/docker/modules/egg/fate_flow ] && rm -r ${source_dir}/docker-build/docker/modules/egg/fate_flow
  [ -d ${source_dir}/docker-build/docker/modules/egg/arch ] && rm -r ${source_dir}/docker-build/docker/modules/egg/arch
  [ -d ${source_dir}/docker-build/docker/modules/egg/federatedml ] && rm -r ${source_dir}/docker-build/docker/modules/egg/federatedml
  [ -d ${source_dir}/docker-build/docker/modules/egg/federatedrec ] && rm -r ${source_dir}/docker-build/docker/modules/egg/federatedrec
  [ -d ${source_dir}/docker-build/docker/modules/python/fate_flow ] && rm -r ${source_dir}/docker-build/docker/modules/python/fate_flow
  [ -d ${source_dir}/docker-build/docker/modules/python/examples ] && rm -r ${source_dir}/docker-build/docker/modules/python/examples
  [ -d ${source_dir}/docker-build/docker/modules/python/arch ] && rm -r ${source_dir}/docker-build/docker/modules/python/arch
  [ -d ${source_dir}/docker-build/docker/modules/python/federatedml ] && rm -r ${source_dir}/docker-build/docker/modules/python/federatedml
  [ -d ${source_dir}/docker-build/docker/modules/python/federatedrec ] && rm -r ${source_dir}/docker-build/docker/modules/python/federatedrec
  [ -d ${source_dir}/docker-build/docker/modules/python/examples ] && rm -r ${source_dir}/docker-build/docker/modules/python/examples
  [ -f ${source_dir}/docker-build/docker/modules/python/eggroll-api-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/python/eggroll-api-*.tar.gz
  [ -d ${source_dir}/docker-build/docker/modules/client/fate_flow ] && rm -r ${source_dir}/docker-build/docker/modules/client/fate_flow
  [ -d ${source_dir}/docker-build/docker/modules/client/examples ] && rm -r ${source_dir}/docker-build/docker/modules/client/examples
  [ -d ${source_dir}/docker-build/docker/modules/client/arch ] && rm -r ${source_dir}/docker-build/docker/modules/client/arch
  [ -d ${source_dir}/docker-build/docker/modules/client/federatedml ] && rm -r ${source_dir}/docker-build/docker/modules/client/federatedml
  [ -d ${source_dir}/docker-build/docker/modules/client/federatedrec ] && rm -r ${source_dir}/docker-build/docker/modules/client/federatedrec
  [ -d ${source_dir}/docker-build/docker/modules/client/examples ] && rm -r ${source_dir}/docker-build/docker/modules/client/examples
  [ -f ${source_dir}/docker-build/docker/modules/client/eggroll-api-*.tar.gz ] && rm ${source_dir}/docker-build/docker/modules/client/eggroll-api-*.tar.gz

  ln ${source_dir}/cluster-deploy/packages/fate-federation-${version}.tar.gz ${source_dir}/docker-build/docker/modules/federation/fate-federation-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/fate-proxy-${version}.tar.gz ${source_dir}/docker-build/docker/modules/proxy/fate-proxy-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/eggroll-roll-${version}.tar.gz ${source_dir}/docker-build/docker/modules/roll/eggroll-roll-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/eggroll-meta-service-${version}.tar.gz ${source_dir}/docker-build/docker/modules/meta-service/eggroll-meta-service-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/fateboard-${fateboard_version}.jar ${source_dir}/docker-build/docker/modules/fateboard/fateboard-${fateboard_version}.jar
  cp -r ${source_dir}/fate_flow ${source_dir}/docker-build/docker/modules/python/fate_flow
  cp -r ${source_dir}/arch ${source_dir}/docker-build/docker/modules/python/arch
  cp -r ${source_dir}/federatedml ${source_dir}/docker-build/docker/modules/python/federatedml
  cp -r ${source_dir}/federatedrec ${source_dir}/docker-build/docker/modules/python/federatedrec
  cp -r ${source_dir}/examples ${source_dir}/docker-build/docker/modules/python/examples
  ln ${source_dir}/cluster-deploy/packages/eggroll-api-${version}.tar.gz ${source_dir}/docker-build/docker/modules/python/eggroll-api-${version}.tar.gz
  cp -r ${source_dir}/fate_flow ${source_dir}/docker-build/docker/modules/client/fate_flow
  cp -r ${source_dir}/arch ${source_dir}/docker-build/docker/modules/client/arch
  cp -r ${source_dir}/federatedml ${source_dir}/docker-build/docker/modules/client/federatedml
  cp -r ${source_dir}/federatedrec ${source_dir}/docker-build/docker/modules/client/federatedrec
  cp -r ${source_dir}/examples ${source_dir}/docker-build/docker/modules/client/examples
  ln ${source_dir}/cluster-deploy/packages/eggroll-api-${version}.tar.gz ${source_dir}/docker-build/docker/modules/client/eggroll-api-${version}.tar.gz
  cp -r ${source_dir}/fate_flow ${source_dir}/docker-build/docker/modules/egg/fate_flow
  cp -r ${source_dir}/arch ${source_dir}/docker-build/docker/modules/egg/arch
  cp -r ${source_dir}/federatedml ${source_dir}/docker-build/docker/modules/egg/federatedml
  cp -r ${source_dir}/federatedrec ${source_dir}/docker-build/docker/modules/egg/federatedrec
  ln ${source_dir}/cluster-deploy/packages/eggroll-api-${version}.tar.gz ${source_dir}/docker-build/docker/modules/egg/eggroll-api-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/eggroll-conf-${version}.tar.gz ${source_dir}/docker-build/docker/modules/egg/eggroll-conf-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/eggroll-computing-${version}.tar.gz ${source_dir}/docker-build/docker/modules/egg/eggroll-computing-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/eggroll-egg-${version}.tar.gz ${source_dir}/docker-build/docker/modules/egg/eggroll-egg-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/eggroll-storage-service-cxx-${version}.tar.gz ${source_dir}/docker-build/docker/modules/egg/eggroll-storage-service-cxx-${version}.tar.gz
  ln ${source_dir}/cluster-deploy/packages/third_party_eggrollv1.tar.gz ${source_dir}/docker-build/docker/modules/egg/third_party_eggrollv1.tar.gz


  # conf
  # egg
  mkdir -p ${source_dir}/docker-build/docker/modules/egg/conf
  cd ${source_dir}/docker-build/docker/modules/egg/
  cp ${source_dir}/eggroll/framework/egg/src/main/resources/egg.properties ./conf
  cp ${source_dir}/eggroll/framework/egg/src/main/resources/log4j2.properties ./conf
  cp ${source_dir}/eggroll/framework/egg/src/main/resources/applicationContext-egg.xml ./conf
  cp ${source_dir}/eggroll/framework/egg/src/main/resources/processor-starter.sh ./conf
  
  # meta-service
  mkdir -p ${source_dir}/docker-build/docker/modules/meta-service/conf
  cd ${source_dir}/docker-build/docker/modules/meta-service/
  cp  ${source_dir}/eggroll/framework/meta-service/src/main/resources/meta-service.properties ./conf
  cp  ${source_dir}/eggroll/framework/meta-service/src/main/resources/log4j2.properties ./conf
  cp  ${source_dir}/eggroll/framework/meta-service/src/main/resources/applicationContext-meta-service.xml ./conf
  
  # roll
  mkdir -p ${source_dir}/docker-build/docker/modules/roll/conf
  cd ${source_dir}/docker-build/docker/modules/roll/
  cp  ${source_dir}/eggroll/framework/roll/src/main/resources/roll.properties ./conf
  cp  ${source_dir}/eggroll/framework/roll/src/main/resources/log4j2.properties ./conf
  cp  ${source_dir}/eggroll/framework/roll/src/main/resources/applicationContext-roll.xml ./conf
  
  # fateboard
  mkdir -p ${source_dir}/docker-build/docker/modules/fateboard/conf
  cd ${source_dir}/docker-build/docker/modules/fateboard/
  touch ./conf/ssh.properties
  cp ${source_dir}/fateboard/src/main/resources/application.properties ./conf
  
  # proxy
  mkdir -p ${source_dir}/docker-build/docker/modules/proxy/conf
  cd ${source_dir}/docker-build/docker/modules/proxy/
  cp ${source_dir}/arch/networking/proxy/src/main/resources/applicationContext-proxy.xml ./conf
  cp ${source_dir}/arch/networking/proxy/src/main/resources/log4j2.properties ./conf
  cp ${source_dir}/arch/networking/proxy/src/main/resources/proxy.properties ./conf
  cp ${source_dir}/arch/networking/proxy/src/main/resources/route_tables/route_table.json ./conf
  
  # federation
  mkdir -p ${source_dir}/docker-build/docker/modules/federation/conf
  cd ${source_dir}/docker-build/docker/modules/federation/
  cp  ${source_dir}/arch/driver/federation/src/main/resources/federation.properties ./conf
  cp  ${source_dir}/arch/driver/federation/src/main/resources/log4j2.properties ./conf
  cp  ${source_dir}/arch/driver/federation/src/main/resources/applicationContext-federation.xml ./conf
  
  # fate_flow
  # federatedml
  # federatedrec
  cd ${source_dir}

  for module in "client" "federation" "proxy" "roll" "meta-service" "fateboard" "egg" "python"
  do
      echo "### START BUILDING ${module} ###"
      docker build --build-arg version=${version} --build-arg fateboard_version=${fateboard_version} --build-arg PREFIX=${PREFIX} --build-arg BASE_TAG=${BASE_TAG} -t ${PREFIX}/${module}:${TAG} -f ${source_dir}/docker-build/docker/modules/${module}/Dockerfile ${source_dir}/docker-build/docker/modules/${module}
      echo "### FINISH BUILDING ${module} ###"
      echo ""
  done;

  rm ${source_dir}/docker-build/docker/modules/federation/fate-federation-${version}.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/federation/conf
  rm ${source_dir}/docker-build/docker/modules/proxy/fate-proxy-${version}.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/proxy/conf
  rm ${source_dir}/docker-build/docker/modules/roll/eggroll-roll-${version}.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/roll/conf
  rm ${source_dir}/docker-build/docker/modules/meta-service/eggroll-meta-service-${version}.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/meta-service/conf
  rm ${source_dir}/docker-build/docker/modules/fateboard/fateboard-${fateboard_version}.jar
  rm -r ${source_dir}/docker-build/docker/modules/fateboard/conf
  rm ${source_dir}/docker-build/docker/modules/egg/eggroll-api-${version}.tar.gz
  rm ${source_dir}/docker-build/docker/modules/egg/eggroll-conf-${version}.tar.gz
  rm ${source_dir}/docker-build/docker/modules/egg/eggroll-computing-${version}.tar.gz
  rm ${source_dir}/docker-build/docker/modules/egg/eggroll-egg-${version}.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/egg/conf
  rm ${source_dir}/docker-build/docker/modules/egg/eggroll-storage-service-cxx-${version}.tar.gz
  rm ${source_dir}/docker-build/docker/modules/egg/third_party_eggrollv1.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/egg/fate_flow
  rm -r ${source_dir}/docker-build/docker/modules/egg/arch
  rm -r ${source_dir}/docker-build/docker/modules/egg/federatedml
  rm -r ${source_dir}/docker-build/docker/modules/egg/federatedrec
  rm -r ${source_dir}/docker-build/docker/modules/python/fate_flow
  rm -r ${source_dir}/docker-build/docker/modules/python/arch
  rm -r ${source_dir}/docker-build/docker/modules/python/federatedml
  rm -r ${source_dir}/docker-build/docker/modules/python/federatedrec
  rm -r ${source_dir}/docker-build/docker/modules/python/examples
  rm ${source_dir}/docker-build/docker/modules/python/eggroll-api-${version}.tar.gz
  rm -r ${source_dir}/docker-build/docker/modules/client/fate_flow
  rm -r ${source_dir}/docker-build/docker/modules/client/arch
  rm -r ${source_dir}/docker-build/docker/modules/client/federatedml
  rm -r ${source_dir}/docker-build/docker/modules/client/federatedrec
  rm -r ${source_dir}/docker-build/docker/modules/client/examples
  rm ${source_dir}/docker-build/docker/modules/client/eggroll-api-${version}.tar.gz
  echo ""
}

pushImage() {
  ## push image
  for module in "federation" "proxy" "roll" "python" "meta-service" "fateboard" "egg" "client"
  do
      echo "### START PUSH ${module} ###"
      docker push ${PREFIX}/${module}:${TAG}
      echo "### FINISH PUSH ${module} ###"
      echo ""
  done;
}

while [ "$1" != "" ]; do
    case $1 in
         package)
                 package
                 ;;
         base)
                 buildBase
                 ;;
         modules)
                 buildModule
                 ;;
         all)
                 package
                 buildBase
                 buildModule
                 ;;
         push)
                pushImage
                ;;
    esac
    shift
done
