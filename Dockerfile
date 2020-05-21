from ubuntu:18.04

# Install base libs
run apt-get update && apt-get install --no-install-recommends -y libpng16-16=1.6.34 \
libtiff5=4.0.9 libjpeg8=8c build-essential=12 wget=1.19.4 git=1:2.17.1 python3.6=3.6.9 python3.6-dev=3.6.9 python3-pip=9.0.1 

# Install python requirements
run pip3 install --user setuptools==46.3.0 wheel==0.34.2 && pip3 install py_trees==0.8.3 networkx==2.2 pygame==1.9.6 \
    six==1.14.0 numpy==1.18.4 psutil==5.7.0 shapely==1.7.0 xmlschema==1.1.3 ephem==3.7.6.0 \
&& mkdir -p /app/scenario_runner

# Install scenario_runner 
add . /app/scenario_runner

# setup environment :
# 
#   CARLA_HOST :    uri for carla package without trailing slash. 
#                   For example, "https://carla-releases.s3.eu-west-3.amazonaws.com/Linux".
#                   If this environment is not passed to docker build, the value
#		    is taken from CARLA_VER file inside the repository.
#
#   CARLA_RELEASE : Name of the package to be used. For example, "CARLA_0.9.9".
#                   If this environment is not passed to docker build, the value
#                   is taken from CARLA_VER file inside the repository.
# 
#
#  It's expected that $(CARLA_HOST)/$(CARLA_RELEASE).tar.gz is a downloadable resource.
#

env CARLA_HOST ""
env CARLA_RELEASE ""

# Extract and install python API and resources from CARLA

run export DEFAULT_CARLA_HOST=$(cat /app/scenario_runner/CARLA_VER|grep HOST|sed 's/HOST\s*=\s*//g') \
&&  export CARLA_HOST=${CARLA_HOST:-$DEFAULT_CARLA_HOST} \
&&  DEFAULT_CARLA_RELEASE=$(cat /app/scenario_runner/CARLA_VER|grep RELEASE|sed 's/RELEASE\s*=\s*//g') \
&&  export CARLA_RELEASE=${CARLA_RELEASE:-$DEFAULT_CARLA_RELEASE} \
&&  echo $CARLA_HOST/$CARLA_RELEASE.tar.gz \
&&  wget -qO- $CARLA_HOST/$CARLA_RELEASE.tar.gz | tar -xzv PythonAPI/carla -C / \
&&  mv /PythonAPI/carla /app/ \
&&  python3 -m easy_install --no-find-links --no-deps $(find /app/carla/ -iname "*py3.*.egg" )


# Setup working environment
workdir /app/scenario_runner
env PYTHONPATH "${PYTHONPATH}:/app/carla/agents:/app/carla"
entrypoint ["/bin/sh" ]
