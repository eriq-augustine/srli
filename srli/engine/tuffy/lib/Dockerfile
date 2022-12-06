FROM postgres:13.7-alpine

ENV POSTGRES_PASSWORD=tuffy

COPY init-tuffy.sh /docker-entrypoint-initdb.d/
RUN chmod 777 /docker-entrypoint-initdb.d/init-tuffy.sh
RUN ls -l /docker-entrypoint-initdb.d/init-tuffy.sh

RUN apk --update add openjdk8-jre

RUN mkdir -p /tuffy/src
RUN \
    cd /tuffy/src \
    && wget 'http://i.stanford.edu/hazy/tuffy/download/tuffy-0.4-july2014.zip' \
    && unzip tuffy-0.4-july2014.zip \
    && cd .. \
    && cp src/tuffy-0.3-jun2014/tuffy.jar .

COPY tuffy.conf /tuffy/
COPY run-tuffy.sh /tuffy/
RUN chmod 777 /tuffy/run-tuffy.sh

ENTRYPOINT /tuffy/run-tuffy.sh
