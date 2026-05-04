# EC2 Deployment Guide

## Prerequisites
- EC2 instance with Python 3.8+
- Security group with port 5000 (or your PORT) open

## Setup Steps

1. **Upload files to EC2**
```bash
scp -r -i your-key.pem ./* ec2-user@your-ec2-ip:~/ml_app/
```

2. **Install dependencies**
```bash
cd ~/ml_app
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Create .env file** (copy from .env.example)
```bash
cp .env.example .env
```

4. **Test locally**
```bash
python app.py
# Should see: "Starting Flask app on 0.0.0.0:5000 (debug=False)"
```

5. **Run with Gunicorn (Production)**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## Using PM2 for Auto-Restart (Optional)
```bash
npm install -g pm2
pm2 start "gunicorn --bind 0.0.0.0:5000 --workers 4 app:app" --name ml_app
pm2 startup
pm2 save
```

## Environment Variables
- `FLASK_DEBUG`: Set to `false` for EC2 (default)
- `HOST`: Bind address (default: `0.0.0.0`)
- `PORT`: Port number (default: `5000`)
- `MODEL_PATH`: Path to model file (default: `brain_tumor_model.h5`)

## Accessing from Browser
- Local: `http://localhost:5000`
- EC2: `http://your-ec2-public-ip:5000`

## Troubleshooting
- Check security group rules allow inbound on port 5000
- Ensure model file exists in the same directory
- Use `sudo` if port < 1024
- Check logs: `tail -f app.log`
