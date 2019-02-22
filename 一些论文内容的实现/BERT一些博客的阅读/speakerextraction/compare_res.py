from honglou import talks
from prediction_results_utf8_1p2million import predictions

for i, talk in enumerate(talks):
    print(talk["context"], " ||| ", predictions["%s"%i])
