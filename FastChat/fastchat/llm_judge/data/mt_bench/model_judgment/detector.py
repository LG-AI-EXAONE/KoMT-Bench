# 2024/08/07: This file has been added by LG AI Research to support the language detection.

import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python import text
import json
from collections import OrderedDict
import math
import jsonlines


def penalty_to_en_answer(model_id,output_path,detector):
  answerkeys={}
  with jsonlines.open(f"../model_answer/{model_id}.jsonl") as f:
    for line in f.iter():
      answerkeys[line["question_id"]]=[line["choices"][0]["turns"][0],line["choices"][0]["turns"][1]]

  with open(f"{output_path}/{model_id}_single_final.jsonl", "w", encoding="utf-8") as e:
    with jsonlines.open(f"./{model_id}_single.jsonl") as f:
      for line in f.iter():
        ori_socre=line["score"]
        turn=line["turn"]
        q_id=line["question_id"]
        lang_result=detector.detect(answerkeys[q_id][turn-1])
        lang_result=lang_result.detections
        line["language"]={}

        for detection in lang_result:
          line["language"][detection.language_code]=f"{detection.probability:.2f}"
          lang_result=detection.language_code
        
        # 영어로 답변 작성시 기존 score에 제곱근을 취합니다.
        # 다만 거진 영어 100% 답변이 허용되는 json 형식의 답변인 138, 140 문항은 패널티 예외 조항으로 합니다
        # 간혹가다 -1 점수가 나오는데 이도 제외합니다.
        if lang_result !="ko" and q_id != 138 and q_id != 140:
          try:
            line["score"]=math.sqrt(ori_socre)
          except:
            pass
        json.dump(line, e, ensure_ascii=False) 
        e.write("\n") 



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_id", required=True,type=str)
  parser.add_argument("--output_path", default=".",type=str)
  parser.add_argument("--detect_model", type=str, default="detector.tflite")
  args = parser.parse_args()
  base_options = python.BaseOptions(model_asset_path=args.detect_model)
  options = text.LanguageDetectorOptions(base_options=base_options,max_results=2)
  detector = text.LanguageDetector.create_from_options(options)
  penalty_to_en_answer(args.model_id,args.output_path,detector)
