FROM python:3.12-slim

#############################
# INSTALL PYTHON DEPENDENCIES
#############################

# install git for pip install git+https://
RUN apt-get -o Acquire::Max-FutureTime=100000 update \
 && apt-get install -y --no-install-recommends build-essential git

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy and install python requirements + ember from github
# copy requirements to /tmp
COPY docker-requirements.txt /tmp/requirements.txt

# install: torch first from CPU index, then the rest (with numpy/signify pins)
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip cache purge

#############################
# REBASE & DEPLOY CODE
#############################

# rebase to make a smaller image
FROM python:3.12-slim

# runtime OS libs needed by lightgbm / torch (OpenMP)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgomp1 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*
 
# copy python virtual env (all dependencies) from previous image
COPY --from=0 /opt/venv /opt/venv

# copy defender code to /opt/defender/defender
COPY defender /opt/defender/defender

#############################
# SETUP ENVIRONMENT
#############################

# open port 8080
EXPOSE 8080

# add a defender user and switch user
RUN groupadd -r defender && useradd --no-log-init -r -g defender defender
USER defender

# change working directory
WORKDIR /opt/defender/

# update environmental variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/defender"
ENV STRICT_EXTRACT="1"

#############################
# RUN CODE
#############################
CMD ["python","-u","-m","defender"]

## TO BUILD IMAGE:
# docker build -t ember .
## TO RUN IMAGE (ENVIRONMENTAL VARIABLES DECLARED ABOVE)
# docker run -itp 8080:8080 ember
## TO RUN IMAGE (OVERRIDE ENVIRONMENTAL VARIABLES DECLARED ABOVE)
# docker run -itp 8080:8080 --env DF_MODEL_GZ_PATH="models/ember_model.txt.gz" --env DF_MODEL_THRESH=0.8336 --env DF_MODEL_NAME=myember ember