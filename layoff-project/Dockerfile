FROM python:3.11

# Set work directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install supervisor

# Expose both ports (but Elastic Beanstalk will use 8080)
EXPOSE 8000 8501 8080

# Copy supervisor config
COPY supervisord.conf /etc/supervisord.conf

# Start both FastAPI and Streamlit
CMD ["supervisord", "-c", "/etc/supervisord.conf"]


