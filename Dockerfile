FROM python:3.8-slim-buster
WORKDIR /app
RUN mkdir seiton

RUN apt-get update 
RUN apt-get install -y python3-pip python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libpangocairo-1.0-0
RUN apt install -y python3-cffi libcairo2 libcairo2-dev libpango-1.0-0 libpango1.0-dev libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
                   libgdk-pixbuf2.0-dev libffi-dev shared-mime-info libffi-dev fonts-font-awesome
COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 uninstall -y weasyprint
RUN pip3 install django-weasyprint
RUN ["chmod", "+x", "pipeline.sh"]
ENTRYPOINT ["./pipeline.sh"]