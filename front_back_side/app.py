from flask import Flask, request, jsonify
import torch
from torch import nn

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 均值和对数方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


# 创建 Flask 应用
app = Flask(__name__)

# 加载预训练的 VAE 模型
vae = VAE(input_dim=10, hidden_dim=64, latent_dim=5)  # 输入和隐状态维度根据需求调整
#vae.load_state_dict(torch.load("vae_model.pth"))  # 确保预训练模型文件在当前目录中
vae.eval()  # 切换到评估模式

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"

@app.route('/generate', methods=['POST'])
def generate_trajectory():
    """
    接收前端传递的隐状态，生成轨迹并返回。
    """
    try:
        # 获取前端传递的隐状态向量
        latent_vector = request.json.get("latent_vector")  # 前端 JSON 中的 "latent_vector"
        latent_vector = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0)

        # 使用解码器生成轨迹
        with torch.no_grad():
            trajectory = vae.decoder(latent_vector).squeeze(0).tolist()
        print("Received latent vector:", latent_vector)
        print("Generated trajectory:", trajectory)
        # 返回轨迹数据
        return jsonify({"trajectory": trajectory})

    except Exception as e:
        # 错误处理
        print('not received')
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)