from rouge_chinese import Rouge
import jieba # you can use any other word cutting library

hypothesis = "患者d天前无明显诱因出现发热，伴有关节疼痛不适，最高体温39.2℃，于外院查血常规示WBC:14.56*10^9/L,N%:87.1%,PLT: 369*10^9/L,ESR:61mm/h,CRP:13.29mg/L,ANA(-),抗SSA抗体(+),抗dsDNA抗体(+)，抗核糖体P蛋白抗体（+），抗心磷脂抗体（-）；为进一步诊治收入我科。患者起病以来，食欲正常，小便正常，大便正常，体重无明显变化。"
hypothesis = ' '.join(jieba.cut(hypothesis))

reference = "患者于 d 天前出现发热，体温最高达 t℃，伴有单个或多个关节疼痛，主要累及膝关节、踝关节还是其他部位。是否伴有晨僵、红肿、活动受限等症状。有无外伤史或风湿性疾病史。"
reference = ' '.join(jieba.cut(reference))

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(scores)
