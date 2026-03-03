@echo off
cd /d "%~dp0"

echo Stopping NexusCortex services...
docker compose down

echo NexusCortex stopped.
echo Data volumes preserved. Use 'docker compose down -v' to remove volumes.
