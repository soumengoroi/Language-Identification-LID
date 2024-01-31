import os
import sys
import tensorflow as tf
import librosa
import json
import numpy as np
import pandas as pd
from vocab.vocab import Vocab

#CREATING THE MODEL AND DEMO RUN
vocab = Vocab("vocab/vocab.txt")
model = tf.saved_model.load('save_model')

#some test audios are provided in test_audios files (chinese.wav, english.wav, french.wav, german.wav, italian.wav, japanese.wav, korean.wav, portuguese.wav,
#russian.wav, spanish.wav, vietnamese), you can specify an audio file to run.
audio_file = 'test_audios/german.wav'     #english.wav'
signal, _ = librosa.load(audio_file, sr=16000)
lang_id, prob = model.predict_pb(signal)
language = vocab.token_list[lang_id.numpy()]
probability = prob.numpy()*100

print("{} is predicted as {} and it's probability={:.2f}% ".format(audio_file, language, probability))
#########################################################################################################
#Spoken_language_identification/test_audios/german.wav is predicted as german and it's probability=99.98% 
#########################################################################################################
def predict_language(path):
  audio_file = path
  signal, _ = librosa.load(audio_file, sr=16000)
  lang_id, prob = model.predict_pb(signal)
  language = vocab.token_list[lang_id.numpy()]
  probability = prob.numpy()*100
  return language,probability

print(predict_language('test_audios/english.wav'))
##################################
# ('english', 99.99750852584839)
##################################

######################language evaluation#################################
test_language_list=['Chinese_China','English','French','German','Indonesian','Italian','Japanese','Portuguese','Russian','Spanish','Turkish','Not_in_list']
#test_language_list=['Chinese_China']
lang_index_dict={test_language_list[i]:i for i in range(len(test_language_list))}
print(lang_index_dict)
print(type(lang_index_dict.get('english')))
#########################################################################
# {'Chinese_China': 0, 'English': 1, 'French': 2, 'German': 3, 'Indonesian': 4, 'Italian': 5, 'Japanese': 6, 'Portuguese': 7, 'Russian': 8, 'Spanish': 9, 'Turkish': 10, 'Not_in_list': 
# 11}
# <class 'NoneType'>
##########################################################################


def get_result(lang_label):
    print('\n\n'+lang_label.upper())

    list=[0 for i in range(len(test_language_list))]
    grand_total_records=0
    grand_total_correct_prediction=0

    #for dev_data
    total_records=0
    correct_prediction=0
    #data=pd.read_csv('/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/dev.csv'.format(lang_label),encoding='utf-16le', sep='\t')
    data=pd.read_csv('dataset/common_voice_kpd/{}/dev.csv'.format(lang_label),encoding='utf-16le', sep='\t')    # dev.csv'.format(lang_label),encoding='windows-1252'utf8)
    #print(data.columns)
    # Clean up column names
    data.columns = data.columns.str.strip()
    #print(data.columns)
    #folder_path='/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/dev'.format(lang_label)#.capitalize())
    folder_path='dataset/common_voice_kpd/{}/dev'.format(lang_label)#.capitalize())
    #print(folder_path)
    for index, row in data.iterrows():
        folder_name = row['client_id']
        #print(folder_name)
        file_name = row['path']
        full_path=folder_path+'/'+folder_name+'/'+file_name;
        language_predicted=predict_language(full_path)[0]
        #print(language_predicted.capitalize())
        language_predicted = language_predicted.capitalize()
        if language_predicted == "Chinese" :
            language_predicted = "Chinese_China"
        if language_predicted in test_language_list:
          list[lang_index_dict.get(language_predicted)]+=1
        else:
          list[11]+=1
        if language_predicted==lang_label:
          correct_prediction+=1
        total_records+=1
    print('folder :: {}\t total_record :: {}\t correct_prediction :: {}\t accuracy :: {} %'.format('dev',total_records,correct_prediction,correct_prediction/total_records*100))

    grand_total_records+=total_records
    grand_total_correct_prediction+=correct_prediction

    #for test_data
    total_records=0
    correct_prediction=0
    #data=pd.read_csv('/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/test.csv'.format(lang_label),encoding='utf-16le', sep='\t')
    data=pd.read_csv('dataset/common_voice_kpd/{}/test.csv'.format(lang_label),encoding='utf-16le', sep='\t')   #'windows-1252')
    # Clean up column names
    data.columns = data.columns.str.strip()
    #folder_path='/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/test'.format(lang_label)#.capitalize())
    folder_path='dataset/common_voice_kpd/{}/test'.format(lang_label)#.capitalize())

    for index, row in data.iterrows():
        folder_name = row['client_id']
        file_name = row['path']
        full_path=folder_path+'/'+folder_name+'/'+file_name;
        language_predicted=predict_language(full_path)[0]
        language_predicted = language_predicted.capitalize()
        if language_predicted == "Chinese" :
            language_predicted = "Chinese_China"
        if language_predicted in test_language_list:
          list[lang_index_dict.get(language_predicted)]+=1
        else:
          list[11]+=1
        if language_predicted==lang_label:
          correct_prediction+=1
        total_records+=1
    print('folder :: {}\t total_record :: {}\t correct_prediction :: {}\t accuracy :: {} %'.format('test',total_records,correct_prediction,correct_prediction/total_records*100))
    grand_total_records+=total_records
    grand_total_correct_prediction+=correct_prediction

    #for train_data
    total_records=0
    correct_prediction=0
    #data=pd.read_csv('/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/train.csv'.format(lang_label),encoding='utf-16le', sep='\t')
    data=pd.read_csv('dataset/common_voice_kpd/{}/train.csv'.format(lang_label),encoding='utf-16le', sep='\t')
    # Clean up column names
    data.columns = data.columns.str.strip()
    folder_path='dataset/common_voice_kpd/{}/train'.format(lang_label)#.capitalize())
    folder_path='dataset/common_voice_kpd/{}/train'.format(lang_label)#.capitalize())

    for index, row in data.iterrows():
        folder_name = row['client_id']
        file_name = row['path']
        full_path=folder_path+'/'+folder_name+'/'+file_name;
        language_predicted=predict_language(full_path)[0]
        language_predicted = language_predicted.capitalize()
        if language_predicted == "Chinese" :
            language_predicted = "Chinese_China"
        if language_predicted in test_language_list:
          list[lang_index_dict.get(language_predicted)]+=1
        else:
          list[11]+=1
        if language_predicted==lang_label:
          correct_prediction+=1
        total_records+=1
    print('folder :: {}\t total_record :: {}\t correct_prediction :: {}\t accuracy :: {} %'.format('train',total_records,correct_prediction,correct_prediction/total_records*100))
    grand_total_records+=total_records
    grand_total_correct_prediction+=correct_prediction


    print('folder :: {}\t total_record :: {}\t correct_prediction :: {}\t accuracy :: {} %'.format('all',grand_total_records,grand_total_correct_prediction,grand_total_correct_prediction/grand_total_records*100))
    return list

accuracy_matrix=[]
accuracy_matrix.append(test_language_list)


#'korean' and 'vietnamese' language is not present in the dataset so we can not evaluate that data
for lang_label in test_language_list[:-1]:
  accuracy_matrix.append(get_result(lang_label))


############################################################################################################################
#   CHINESE_CHINA
# folder :: dev	 total_record :: 98	 correct_prediction :: 73	 accuracy :: 74.48979591836735 %
# folder :: test	 total_record :: 100	 correct_prediction :: 78	 accuracy :: 78.0 %
# folder :: train	 total_record :: 401	 correct_prediction :: 313	 accuracy :: 78.05486284289277 %
# folder :: all	 total_record :: 599	 correct_prediction :: 464	 accuracy :: 77.46243739565944 %


# ENGLISH
# folder :: dev	 total_record :: 79	 correct_prediction :: 23	 accuracy :: 29.11392405063291 %
# folder :: test	 total_record :: 98	 correct_prediction :: 20	 accuracy :: 20.408163265306122 %
# folder :: train	 total_record :: 414	 correct_prediction :: 94	 accuracy :: 22.705314009661837 %
# folder :: all	 total_record :: 591	 correct_prediction :: 137	 accuracy :: 23.181049069373945 %


# FRENCH
# folder :: dev	 total_record :: 111	 correct_prediction :: 67	 accuracy :: 60.36036036036037 %
# folder :: test	 total_record :: 106	 correct_prediction :: 56	 accuracy :: 52.83018867924528 %
# folder :: train	 total_record :: 405	 correct_prediction :: 185	 accuracy :: 45.67901234567901 %
# folder :: all	 total_record :: 622	 correct_prediction :: 308	 accuracy :: 49.51768488745981 %


# GERMAN
# folder :: dev	 total_record :: 97	 correct_prediction :: 67	 accuracy :: 69.0721649484536 %
# folder :: test	 total_record :: 101	 correct_prediction :: 70	 accuracy :: 69.3069306930693 %
# folder :: train	 total_record :: 413	 correct_prediction :: 239	 accuracy :: 57.869249394673126 %
# folder :: all	 total_record :: 611	 correct_prediction :: 376	 accuracy :: 61.53846153846154 %


# INDONESIAN
# folder :: dev	 total_record :: 166	 correct_prediction :: 122	 accuracy :: 73.49397590361446 %
# folder :: test	 total_record :: 150	 correct_prediction :: 71	 accuracy :: 47.333333333333336 %
# folder :: train	 total_record :: 595	 correct_prediction :: 331	 accuracy :: 55.63025210084034 %
# folder :: all	 total_record :: 911	 correct_prediction :: 524	 accuracy :: 57.5192096597146 %


# ITALIAN
# folder :: dev	 total_record :: 92	 correct_prediction :: 73	 accuracy :: 79.34782608695652 %
# folder :: test	 total_record :: 100	 correct_prediction :: 73	 accuracy :: 73.0 %
# folder :: train	 total_record :: 373	 correct_prediction :: 278	 accuracy :: 74.53083109919572 %
# folder :: all	 total_record :: 565	 correct_prediction :: 424	 accuracy :: 75.04424778761062 %


# JAPANESE
# folder :: dev	 total_record :: 122	 correct_prediction :: 95	 accuracy :: 77.8688524590164 %
# folder :: test	 total_record :: 144	 correct_prediction :: 106	 accuracy :: 73.61111111111111 %
# folder :: train	 total_record :: 490	 correct_prediction :: 315	 accuracy :: 64.28571428571429 %
# folder :: all	 total_record :: 756	 correct_prediction :: 516	 accuracy :: 68.25396825396825 %


# PORTUGUESE
# folder :: dev	 total_record :: 130	 correct_prediction :: 37	 accuracy :: 28.46153846153846 %
# folder :: test	 total_record :: 124	 correct_prediction :: 48	 accuracy :: 38.70967741935484 %
# folder :: train	 total_record :: 492	 correct_prediction :: 156	 accuracy :: 31.70731707317073 %
# folder :: all	 total_record :: 746	 correct_prediction :: 241	 accuracy :: 32.30563002680965 %


# RUSSIAN
# folder :: dev	 total_record :: 98	 correct_prediction :: 70	 accuracy :: 71.42857142857143 %
# folder :: test	 total_record :: 100	 correct_prediction :: 71	 accuracy :: 71.0 %
# folder :: train	 total_record :: 388	 correct_prediction :: 217	 accuracy :: 55.927835051546396 %
# folder :: all	 total_record :: 586	 correct_prediction :: 358	 accuracy :: 61.092150170648466 %


# SPANISH
# folder :: dev	 total_record :: 96	 correct_prediction :: 34	 accuracy :: 35.41666666666667 %
# folder :: test	 total_record :: 94	 correct_prediction :: 36	 accuracy :: 38.297872340425535 %
# folder :: train	 total_record :: 389	 correct_prediction :: 112	 accuracy :: 28.79177377892031 %
# folder :: all	 total_record :: 579	 correct_prediction :: 182	 accuracy :: 31.43350604490501 %


# TURKISH
# folder :: dev	 total_record :: 140	 correct_prediction :: 88	 accuracy :: 62.857142857142854 %
# folder :: test	 total_record :: 137	 correct_prediction :: 91	 accuracy :: 66.42335766423358 %
# folder :: train	 total_record :: 547	 correct_prediction :: 341	 accuracy :: 62.3400365630713 %
# folder :: all	 total_record :: 824	 correct_prediction :: 520	 accuracy :: 63.10679611650486 %
#################################################################################################################


import pandas as pd
accuracy_matrix_df=pd.DataFrame(data=accuracy_matrix[1:],columns=accuracy_matrix[0])
from IPython.display import display
display(accuracy_matrix_df)


###############################################################################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#################################  Duration wish audio Distrubution ###########################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
def get_path(lang_label):
    print('\n\n'+lang_label.upper())

    path_list=[]
    super_folder_list=['dev','train','test']
    #for dev_data
    for super_folder in super_folder_list:
        #data=pd.read_csv('/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/{}.csv'.format(lang_label,super_folder),encoding='utf-16le', sep='\t')
        data=pd.read_csv('dataset/common_voice_kpd/{}/{}.csv'.format(lang_label,super_folder),encoding='utf-16le', sep='\t')  #encoding='windows-1252')
        # Clean up column names
        data.columns = data.columns.str.strip()
        #folder_path='/home/soumen/LID/GitHub/dataset/common_voice_kpd/{}/{}'.format(lang_label,super_folder)
        folder_path='dataset/common_voice_kpd/{}/{}'.format(lang_label,super_folder)  #.capitalize(),super_folder)
        for index, row in data.iterrows():
            folder_name = row['client_id']
            file_name = row['path']
            full_path=folder_path+'/'+folder_name+'/'+file_name;
            path_list.append(full_path)
    return path_list


sr=16000
def process_audio(path,lang_label,count_dict,correct_dict):
    audio,_=librosa.load(path,sr=sr)
    duration=len(audio)/sr
    lang_id, prob = model.predict_pb(audio)
    language = vocab.token_list[lang_id.numpy()]
    language = language.capitalize()
#     print(language)
#     print(lang_label)
#     print("*")
    if language == "Chinese":
        language = "Chinese_China"
    #time to categorize
    label=""

    if(duration>=0 and duration<3):
        label='(0,3)'
    elif(duration>=3 and duration<6):
        label='(3,6)'
    elif(duration>=6 and duration<9):
        label='(6,9)'
    elif(duration>=9 and duration<12):
        label='(9,12)'
    elif(duration>=12 and duration<15):
        label='(12,15)'
    elif(duration>=15 and duration<18):
        label='(15,18)'
    else:
        return

    count_dict[label]+=1
    # print(language)
    if(language==lang_label):
        correct_dict[label]+=1


#the time range is (0-3) (3-6) (6,9) (9,12) (12,15) (15,18)
def get_range_stat(lang_label):
    lang_path_list=get_path(lang_label)
    range_correct_predict_dict={
        '(0,3)' : 0,
        '(3,6)' : 0,
        '(6,9)' : 0,
        '(9,12)' : 0,
        '(12,15)' : 0,
        '(15,18)' : 0
    }
    range_total_count_dict={
        '(0,3)' : 0,
        '(3,6)' : 0,
        '(6,9)' : 0,
        '(9,12)' : 0,
        '(12,15)' : 0,
        '(15,18)' : 0
    }
    for path in lang_path_list:
        process_audio(path,lang_label,range_total_count_dict,range_correct_predict_dict)
    return range_total_count_dict,range_correct_predict_dict

def get_accuracy(count_dict,predict_dict):
    # for escaping from error we are setting all 0 into 1
    for label in count_dict.keys():
        if count_dict.get(label)==0:
            count_dict[label]=-1
    accuracy_dict={label : predict_dict.get(label)/count_dict.get(label)*100 for label in count_dict.keys()}
    for label in count_dict.keys():
        if count_dict.get(label)==-1:
            count_dict[label]=0
    return accuracy_dict

import matplotlib.pyplot as plt
def plot_barchart(lang_label,total_count_dict,correct_predict_dict):
    range_labels = total_count_dict.keys()
    count_list = [total_count_dict.get(label) for label in range_labels ]
    correct_list = [correct_predict_dict.get(label) for label in range_labels ]

    # Position of the bars on the x-axis
    x_pos_count = range(len(range_labels))
    x_pos_correct = [pos + 0.3 for pos in x_pos_count]  # Shift the bars for 2024 to the right

    # Create the bar chart

    plt.bar(x_pos_count, count_list, width=0.3, label='count')
    plt.bar(x_pos_correct, correct_list, width=0.3, label='correct')

    # Add labels, title, and legend
    plt.xlabel('audio file length')
    plt.ylabel('statistics')
    plt.title(lang_label)
    plt.xticks([pos + 0.15 for pos in x_pos_count], range_labels)
    plt.legend()

    # Add marks as text labels above each bar
    for i, v in enumerate(count_list):
        plt.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')

    for i, v in enumerate(correct_list):
        plt.text(i + 0.3, v + 2, str(v), ha='center', va='bottom', fontweight='bold')

    # Show the plot
    plt.show()


import pandas as pd
from IPython.display import display
def get_accuracy_table(lang_label,total_count_dict,correct_predict_dict,accuracy_dict):
    total_count=0
    correct_prediction=0
    column_names=['duration','count','correct','accuracy']
    rows=[]
    for label in total_count_dict.keys():
        total_count+=total_count_dict.get(label)
        correct_prediction+=correct_predict_dict.get(label)
        temp=[label,total_count_dict.get(label),correct_predict_dict.get(label),'{:.2f}%'.format(accuracy_dict.get(label))]
        rows.append(temp)
    df = pd.DataFrame(data=rows, columns=column_names)
    display(df)
    print('overall accuracy :: {:.2f}'.format(correct_prediction/total_count*100))

def get_language_result(lang_label):
    total_count_dict,correct_predict_dict=get_range_stat(lang_label)
    accuracy_dict=get_accuracy(total_count_dict,correct_predict_dict)
    plot_barchart(lang_label,total_count_dict,correct_predict_dict)
    get_accuracy_table(lang_label,total_count_dict,correct_predict_dict,accuracy_dict)


########################### Put language for result #############################
get_language_result('Chinese_China')
get_language_result('English')
