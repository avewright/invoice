# 1) Ensure venv and install deps
python -m venv /root/invoice/.venv
source /root/invoice/.venv/bin/activate
pip install --upgrade pip
pip install -r /root/invoice/requirements.txt

# 2) Fix permissions and run
chmod +x /root/invoice/run.sh
PORT=8000 /root/invoice/run.sh