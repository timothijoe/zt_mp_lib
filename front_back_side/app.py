from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch import nn

import sys_path_utils 
from vae_train.traj_vae import VaeEncoder, VaeDecoder
import numpy as np
device = 'cpu'
device = 'cuda'
decoder_state_dict = torch.load('/home/zhoutong/dir_sda/betty/zt_mp_lib/result/zt_jan13_024/ckpt/150_decoder_ckpt',map_location=torch.device(device))
vae_decoder = VaeDecoder(
    embedding_dim = 64,
    h_dim = 128,
    latent_dim = 5,
    seq_len = 30,
    dt = 0.1,
    device = device,
)
vae_decoder.load_state_dict(decoder_state_dict)
vae_decoder.to(device)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

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

        init_state = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        init_state = torch.from_numpy(init_state).to(torch.float32)
        init_state = init_state.unsqueeze(0)
        init_state = init_state.to(device)
        z = latent_vector.to(device)


        # 使用解码器生成轨迹
        with torch.no_grad():
            traj = vae_decoder(z, init_state)

        init_state = init_state[:, :4]
        traj = traj[:,:, :4]
        traj = torch.cat([init_state.unsqueeze(1), traj], dim = 1)
        traj = traj[0,:,:2]
        traj_cpu = traj.detach().to('cpu').numpy()
        traj_cpu = traj_cpu.tolist()


        print("Received latent vector:", latent_vector)
        # print("Generated trajectory:", trajectory)
        # 返回轨迹数据
        return jsonify({"trajectory": traj_cpu})

    except Exception as e:
        # 错误处理
        print('not received')
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)