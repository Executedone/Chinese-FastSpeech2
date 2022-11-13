''' 
_*_ coding: utf-8 _*_
Date: 2022/3/4
Author: 
Intent:
'''

from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import pywsgi
import torch

import logging, datetime, os, json, yaml
import logging.handlers

from synthesize_all import SpeechSynthesis


class TTSServer(object):
    def __init__(self, config):
        self.config = config
        self.logger = self.set_logger()
        if torch.cuda.is_available() and int(config['gpu_id']) >= 0:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
        self.tts = SpeechSynthesis("./config/AISHELL3")
        self.logger.info("tts engine initialized completely...")

    def set_logger(self):
        logger = logging.getLogger('tts_logger')
        logger.setLevel(logging.INFO)
        log_dir = self.config['log_dir']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        all_log = os.path.join(log_dir, 'all.log')
        error_log = os.path.join(log_dir, 'error.log')
        rf_h = logging.handlers.TimedRotatingFileHandler(all_log,
                                                         when='midnight',
                                                         interval=1,
                                                         backupCount=7,
                                                         atTime=datetime.time(0,0,0,0))
        rf_h.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s'))
        f_h = logging.FileHandler(error_log)
        f_h.setLevel(logging.ERROR)
        f_h.setFormatter(
                logging.Formatter('%(asctime)s - %(filename)s[:%(lineno)d] - %(levelname)s - %(message)s')
        )
        s_h = logging.StreamHandler(stream=None)
        s_h.setLevel(logging.INFO)
        s_h.setFormatter(logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s'))

        logger.addHandler(rf_h)
        logger.addHandler(f_h)
        logger.addHandler(s_h)
        return logger

    def start_server(self):
        app = Flask(__name__)

        @app.route('/test')
        def test():
            return 'Connection Success!'

        @app.route('/TextToSpeech', methods=['GET', 'POST'])
        def synth():
            res = {}
            try:
                if request.method == 'POST':
                    text = request.form.get('text')
                    try:
                        pitch_control = float(request.form.get('pitch_control'))
                    except:
                        pitch_control = 1.0
                    try:
                        energy_control = float(request.form.get('energy_control'))
                    except:
                        energy_control = 1.0
                    try:
                        duration_control = round(1.0 / float(request.form.get('duration_control')), 1)
                    except:
                        duration_control = 1.0
                else:
                    text = request.args.get('text')
                    try:
                        pitch_control = float(request.args.get('pitch_control'))
                    except:
                        pitch_control = 1.0
                    try:
                        energy_control = float(request.args.get('energy_control'))
                    except:
                        energy_control = 1.0
                    try:
                        duration_control = round(1.0 / float(request.args.get('duration_control')), 1)
                    except:
                        duration_control = 1.0

                self.logger.info(f'输入文本：{text}')
                wav_path = self.tts.text2speech(text, pitch_control, energy_control, duration_control)

                res['result'] = wav_path
                res['success'] = True
                res['message'] = 'succeed!'

            except Exception as e:
                self.logger.error(str(e))
                res['result'] = None
                res['success'] = False
                res['message'] = str(e)

            return json.dumps(res, ensure_ascii=False)

        server = pywsgi.WSGIServer((str(self.config['http_id']), self.config['port']), app)
        server.serve_forever()


if __name__ == "__main__":
    server_config = yaml.load(open("./config/AISHELL3/server.yaml", "r"), Loader=yaml.FullLoader)
    server = TTSServer(server_config)
    server.start_server()

    pass