# Group-QPP
This is the github repository of paper "Groupwise Query Performance Prediction with BERT" in ECIR2022.

You can find the Groupwise QPP model(COBERT) and Vanilla BERT model for QPP here.

You may need to prepare the previous baseline files properly as the paper demonstrates.

You can run the pipeline_datasetname.sh to get the results, but before running the shell, you should carefully config the parameters to make sure they're right.
 The data processing part is kind of complicate and I will make a more detailed instruction later.

You need to tokenize your data into a .csv file, formed as **"topic_id, docid, bias(we use maxp in our experiments,this shows the position of the passage in its long document, not used in our main experiments), input_id, attention_mask, segment_mask, qrel_score(0 or 1)"**, one record per line, and different data fields are separated by commas.

And other files like **BERT checkpoints, qrel files, topic id list (one id per line), and initial QPP file (formed as "topic_id score")** are also needed in our experiments.

By the way, you don't need to split the id into train and test for 30 random rounds by yourself. Our code will do it for you.
