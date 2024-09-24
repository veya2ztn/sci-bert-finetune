for i in {0..50};
do 
#nohup python clear_unArxive_data.py --id ${i} > log/processing.${i}.log&
#nohup python script/postprocess.openai_embedding_to_numpy.py --id ${i} > log/processing.${i}.log&
nohup python script/create_answer_tokens.py --id ${i} > log/processing.${i}.log&
done