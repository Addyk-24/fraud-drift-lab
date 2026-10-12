FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit plotly pandas pyarrow numpy
COPY app ./app
COPY results ./results
COPY .streamlit ./.streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
