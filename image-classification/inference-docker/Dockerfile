ARG NUCLIO_LABEL=1.1.18
ARG NUCLIO_ARCH=amd64
ARG NUCLIO_BASE_IMAGE=tensorflow/tensorflow:1.13.2-py3
ARG NUCLIO_ONBUILD_IMAGE=quay.io/nuclio/handler-builder-python-onbuild:${NUCLIO_LABEL}-${NUCLIO_ARCH}

# Supplies processor uhttpc, used for healthcheck
FROM nuclio/uhttpc:0.0.1-amd64 as uhttpc

# Supplies processor binary, wrapper
FROM ${NUCLIO_ONBUILD_IMAGE} as processor

# From the base image
FROM ${NUCLIO_BASE_IMAGE}

# Copy required objects from the suppliers
COPY --from=processor /home/nuclio/bin/processor /usr/local/bin/processor
COPY --from=processor /home/nuclio/bin/py /opt/nuclio/
COPY --from=uhttpc /home/nuclio/bin/uhttpc /usr/local/bin/uhttpc

RUN pip install nuclio-sdk msgpack --no-index --find-links /opt/nuclio/whl
RUN pip install numpy keras requests pillow

RUN apt-get update && apt-get install -y git

# USER CONTENT
ADD ./function /opt/nuclio

# END OF USER CONTENT

# Run processor with configuration and platform configuration
CMD [ "processor" ]