import pandas as pd

data1 = pd.read_csv("data/unarXive_quantum_physics/query.question.results.good_questions1.csv")
data2 = pd.read_csv("data/unarXive_quantum_physics/query.question.results.good_questions2.csv")
data  = pd.concat([data1,data2])
data.to_csv("data/unarXive_quantum_physics/query.question.results.good_questions.csv")
