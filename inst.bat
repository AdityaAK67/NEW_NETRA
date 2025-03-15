@echo off
echo Installing required dependencies...
pip install inference==0.36.1 ^
    inference-cli==0.9.12rc1 ^
    inference-gpu==0.9.12rc1 ^
    boto3==1.35.60 ^
    botocore==1.35.60 ^
    s3transfer==0.10.0 ^
    opencv-python==4.10.0.84 ^
    requests==2.32.0 ^
    docker==7.0.0 ^
    numpy==1.26.4 ^
    skypilot==0.4.1
echo Installation complete!
pause
