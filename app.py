from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# 加载模型
model = joblib.load("gbdt_model.pkl")

# 模型特征顺序
top_features = ['AST', 'GGT', 'AFP', 'AGR']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = [float(data[feat]) for feat in top_features]
        input_array = np.array(input_data).reshape(1, -1)

        # 模型预测概率
        proba = model.predict_proba(input_array)[0]
        score = proba[1]

        # 风险等级
        if score >= 0.510:
            grade = "High risk"
        elif score < 0.415:
            grade = "Low risk"
        else:
            grade = "Medium risk"

        return jsonify({
            "score": score,
            "grade": grade
        })

    except Exception as e:
        return jsonify({"error": str(e)})
# 主页路由
import os
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # 默认端口为 5001
    app.run(host='0.0.0.0', port=port, debug=True)
