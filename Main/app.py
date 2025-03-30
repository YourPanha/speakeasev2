from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run-model', methods=['GET'])
def run_model():
    try:
        # Run the Python model script
        process = subprocess.Popen(["python", "C:/Users/Rakesh/OneDrive/Desktop/Codes/AceHack/speakease/ML/final_pred.py"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if process.returncode == 0:
            return jsonify({"message": "Model ran successfully!", "output": output.decode()})
        else:
            return jsonify({"error": error.decode()}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
