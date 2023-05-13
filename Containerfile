FROM python:3.10-alpine
COPY . /app
WORKDIR /app
RUN pip install .
CMD ["infer_subc"]
# untested
