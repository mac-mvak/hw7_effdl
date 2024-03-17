FROM python:3.11

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY requirements_torch.txt .
RUN pip3 install -r requirements_torch.txt


RUN mkdir /app
WORKDIR /app

COPY proto /app/proto
COPY run_codegen.py /app
RUN python3 run_codegen.py

COPY flask_server.py /app
COPY grpc_server.py /app
COPY downloader.py /app
RUN python3 downloader.py

COPY two_servers.sh /app
RUN chmod a+x two_servers.sh

ENTRYPOINT ["./two_servers.sh"]
