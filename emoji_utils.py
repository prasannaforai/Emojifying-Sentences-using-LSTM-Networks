import numpy as np
import pandas as pd
import emoji
import matplotlib.pyplot as plt


# Function to read data from file
def read_csv_file(filename):
    
    data = pd.read_csv(filename)
    X = np.asarray(data['text'])
    Y = np.asarray(data['emoji'], dtype=int)
    return X, Y

# Dict of emoji's we are going to use
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

# Converts labels(0,..4) into emoji's
def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

# Function to convert lables to one-hot vectors
def one_hot(Y, num_classes):
    Y = np.eye(num_classes)[Y.reshape(-1)]
    return Y

# Function to read the glove_data_set of vectors
def read_glove_vecs(glove_file):
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        
        # In each line the first element is the word and remaining other elements are vectors.
        for line in f:
            line = line.strip().split()
            current_word = line[0]
            words.add(current_word)
            word_to_vec_map[current_word] = np.array(line[1:], dtype=np.float64)
    
        idx = 1
        words_to_index = {}
        index_to_words = {}
        for word in sorted(words):
            words_to_index[word] = idx
            index_to_words[idx] = word
            idx += 1
            
    return words_to_index, index_to_words, word_to_vec_map

# Softmax function 
def softmax(z):
    e_x = np.exp(z - np.max(z))
    return e_x/sum(e_x)

def predict(X, Y, W, b, word_to_vec_map):
    
    m = X.shape[0]
    preds = np.zeros((m, 1))
    
    avg = np.zeros((50, ))
    
    for i in range(m):
        
        words = list(map(lambda word : word.lower(), X[i].split()))
        
        for word in words:
            avg += word_to_vec_map[word]
        avg = avg/len(words)
        
        Z = np.matmul(W, avg) + b
        probs = softmax(Z)
        
        preds[i] = np.argmax(probs)
        
    print("Accuracy : %f"%(np.mean(preds[:] == Y.reshape(Y.shape[0], 1)[:])))
    
    return preds


# Function to print predictions
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        
        
def plot_confusion_matrix(Y_true, Y_pred, title='Confusion Matrix', cmap=plt.cm.gray):
    
    df_confusion = pd.crosstab(Y_true, Y_pred.reshape(Y_pred.shape[0], ), rownames=['TRUE'], colnames=['PREDICTED'],
                              margins=True)
    
    df_confusion_norm = df_confusion/df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
def print_mislabelled_sentences(X, Y_true, Y_pred):
    
    mislabelled_indices = []
    
    for i in range(X.shape[0]):
        if int(Y_true.flatten()[i] == Y_pred.flatten()[i]) != 1:
            print("   Index  :", i)
            print("   True   :", X[i], label_to_emoji(int(Y_true[i])))
            print("Predicted :", X[i], label_to_emoji(int(Y_pred[i])))
            print("---------------------------------------")
            mislabelled_indices.append(i)
               
    return mislabelled_indices         