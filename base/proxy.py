import os
# # 验证设置是否成功
def set_proxy():
    os.environ["http_proxy"]="http://localhost:8890"
    os.environ["https_proxy"]="http://localhost:8890"
    os.environ["WANDB_DISABLED"] = "true"