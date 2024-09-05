# Done by:
# Tan Hong Guang Melvin, A0223820R
# Tan Hui Rong, A0216246J
# Myo Nyi Nyi A0239155B

## Q2.1(a) ### --------------------------------------------------------------------------------
from collections import defaultdict, Counter
def naive_ouput(filename, output_filename, delta=0.1):
    with open(filename, 'r', encoding = 'utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip() != '']
    
    processed_lines = []

    # handle tokens and tags, converting tokens to lowercase if not a URL
    for line in lines:
        token, tag = line.split('\t')
        if 'http' not in token:
            token = token.lower()
        processed_lines.append((token, tag))

    # count occurrences of each token-tag pair and each tag using Counter

    token_tag_counter = Counter(processed_lines)
    tag_counter = Counter(tag for _, tag in processed_lines)
    
    # unique token for smoothing
    unique_tokens = set(token for token, _ in processed_lines)
    num_words = len(unique_tokens)
    
    delta = 0.1  # smoothing parameter
    
    # probabilities with smoothing for each token-tag pair
    probabilities = {}
    for (token, tag), count in token_tag_counter.items():
        prob = (count + delta) / (tag_counter[tag] + delta * (num_words + 1))
        probabilities[(token, tag)] = prob
    
    # handle unseen tokens
    for tag in tag_counter:
        probabilities[('WordNotFound', tag)] = delta / (tag_counter[tag] + delta * (num_words + 1))
    
    with open(output_filename, 'w', encoding= 'utf-8') as file:
        for (token, tag), prob in probabilities.items():
            file.write(f"{token}\t{tag}\t{prob}\n")

## Q2.1(b) ### --------------------------------------------------------------------------------

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    
    # nested dictionary to hold the probabilities, with a default value of negative infinity
    probabilities = defaultdict(lambda: defaultdict(lambda: float('-inf')))

    with open(in_output_probs_filename, 'r', encoding="utf-8") as file:
        for line in file:
            token, tag, prob = line.strip().split('\t')
            probabilities[token][tag] = float(prob)
    
    # tokens without their associated POS tags.
    with open(in_test_filename, 'r', encoding="utf-8") as file:
        test_data = file.readlines()
    
    predictions = []
    for line in test_data:
        if line.strip():  
            token = line.strip().lower()
            if 'http' not in token:  
                token = token.lower()

            # find best tag for the token by maximizing the probability. if the token is not found,
            # use the 'WordNotFound' default probability. This is equivalent to calculating argmax P(y = j|x = w).

            best_tag = max(probabilities[token], key=probabilities[token].get, default='WordNotFound')
            predictions.append(best_tag)
        else:  # preserve sentence boundaries
            predictions.append('')
    
    with open(out_prediction_filename, 'w', encoding="utf-8") as file:
        file.write('\n'.join(predictions) + '\n')

## Q2.1(c) ### --------------------------------------------------------------------------------
# Naive prediction accuracy: 936/1378 = 0.6792452830188679

## Q2.2(a) ### --------------------------------------------------------------------------------
# P(x = w| y = j ) * P(y = j) / P(x = w)

## Q2.2(b) ### --------------------------------------------------------------------------------

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    
    #dictionary to load output probabilities, setting a very low default to signify unlikelihood
    output_probabilities = defaultdict(lambda: defaultdict(lambda: float('-inf')))

    with open(in_output_probs_filename, 'r', encoding="utf-8") as file:
        for line in file:
            token, tag, prob = line.strip().split('\t')
            output_probabilities[token][tag] = float(prob)

    # count occurrences of each tag in the training data to calculate P(y = j).
    tag_counter = Counter()
    total_tags = 0
    
    with open(in_train_filename, 'r', encoding="utf-8") as file:
        for line in file:
            if line.strip():
                _, tag = line.strip().split('\t')
                tag_counter[tag] += 1
                total_tags += 1
    
    #P(y = j) for each tag
    tag_probabilities = {tag: count / total_tags for tag, count in tag_counter.items()}
    
    with open(in_test_filename, 'r', encoding="utf-8") as file:
        test_data = file.readlines()
    
    predictions = []
    # for each token in the test data, predict the most probable tag.
    for line in test_data:
        if line.strip():
            token = line.strip().lower()
            if 'http' not in token:
                token = token.lower()
            
            # calculate P(y=j | x=w) = P(x = w| y = j) * P(y = j) for each tag and find the best tag
            # since P(x=w) is constant across tags, it's omitted from the calculation

            best_tag = max(tag_probabilities.keys(), key=lambda tag: (output_probabilities[token][tag] if token in output_probabilities else output_probabilities['WordNotFound'][tag]) * tag_probabilities[tag])
            predictions.append(best_tag)
        else:  # preserve sentence boundaries in the prediction output.
            predictions.append('')
    
    with open(out_prediction_filename, 'w', encoding="utf-8") as file:
        file.write('\n'.join(predictions) + '\n')

## Q2.2(c) ### --------------------------------------------------------------------------------
# Naive2 prediction accuracy: 986/1378 = 0.7155297532656023

## Q3 (a) ### --------------------------------------------------------------------------------
import json
def compute_mle_probs(train_filename, output_probs_filename, trans_probs_filename, delta=0.1):
    # Initialize dictionaries to store counts
    output_counts = defaultdict(lambda: defaultdict(int))
    transition_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    total_words = set()

    # Read training data and compute counts
    with open(train_filename, 'r', errors='ignore') as train_file:
        previous_tag = None
        for line in train_file:
            if line.strip():  # Non-empty line
                token, tag = line.strip().split()
                total_words.add(token)
                output_counts[tag][token] += 1
                tag_counts[tag] += 1
                if previous_tag:
                    transition_counts[previous_tag][tag] += 1
                previous_tag = tag
            else:  # Empty line indicates end of a sentence
                previous_tag = None
    
    # Compute output probabilities
    with open(output_probs_filename, 'w', errors='ignore') as output_probs_file:
        for tag, token_counts in output_counts.items():
            total_count = sum(token_counts.values()) # for each tag in output counts, there is dict of keys for each token and their counts
            # this only shows number of times the tag appears
            for token, count in token_counts.items():
                prob = (count + delta) / (total_count + delta * (len(total_words)+ 1))  # Add-one smoothing, edit?
                output_probs_file.write(f"{token} {tag} {prob}\n") 
                ## tag to token : y(tag) -> x(word)
            unseen_p = (0 + delta) / (total_count + delta * (len(total_words)+ 1))
            output_probs_file.write(f"UNSEEN_TEST {tag} {unseen_p}\n") 
    
    # Compute transition probabilities
    with open(trans_probs_filename, 'w',  errors='ignore') as trans_probs_file:
        for prev_tag, next_tag_counts in transition_counts.items():
            total_count = sum(next_tag_counts.values())
            for next_tag, count in next_tag_counts.items():
                prob = (count + delta) / (total_count + delta * len(transition_counts))  # smoothing, len(transition counts) should give total number of tags 
                # some tags dont transit to other to other tags, so you need to denom all num of tags
                trans_probs_file.write(f"{prev_tag} {next_tag} {prob}\n")
                ## tag1 to tag2 : y1(tag1) -> y2(tag2)

## Q3 (b) ### --------------------------------------------------------------------------------
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    # Load tags
    with open(in_tags_filename, 'r', errors='ignore') as tags_file:
        tags = tags_file.read().splitlines()

    # Load transition probabilities
    trans_probs = {}
    with open(in_trans_probs_filename, 'r', errors='ignore') as trans_probs_file:
        for line in trans_probs_file:
            prev_tag, next_tag, prob = line.strip().split()
            if prev_tag not in trans_probs:
                trans_probs[prev_tag] = {}
            trans_probs[prev_tag][next_tag] = float(prob)

    output_probs = {}
    with open(in_output_probs_filename, 'r', errors='ignore') as output_probs_file:
        for line in output_probs_file:
            token, tag, prob = line.strip().split()
            if tag not in output_probs:
                output_probs[tag] = {}
            output_probs[tag][token] = float(prob)

    for tag, token_probs in output_probs.items():
        total_prob = sum(token_probs.values())
        num_tokens = len(token_probs)
        average_prob = total_prob / num_tokens

    # Process each sentence in the test file and write predictions to the output file
    with open(in_test_filename, 'r', errors='ignore') as test_file, \
        open(out_predictions_filename, 'w', errors='ignore') as predictions_file:
        sentence_tokens = []

        for line in test_file:
            # Process each line of the test file
            if line.strip():
                tokens = line.strip().split()
                sentence_tokens.extend(tokens)
            else:
                # Call the Viterbi algorithm for each sentence and write predictions
                p_stateseq, best_tag_sequence = viterbi(sentence_tokens, tags, trans_probs, output_probs)
                for token, tag in zip(sentence_tokens, best_tag_sequence):
                    predictions_file.write(f"{tag}\n")
                predictions_file.write("\n")  # Empty line between sentences
                sentence_tokens = []
                # break ### TEST FOR 1 SENTENCE

        # Process the last sentence if any
        if sentence_tokens:
            best_tag_sequence = viterbi(sentence_tokens, tags, trans_probs, output_probs)
            for token, tag in zip(sentence_tokens, best_tag_sequence):
                predictions_file.write(f"{tag}\n")


# The Viterbi algorithm for Hidden Markov Models
def viterbi(obs, states, trans_p, output_p):
    # Compute total number of transitions
    total_transitions = sum(len(next_tag_dict) for next_tag_dict in trans_p.values())
    # Initialize start_p as a regular dictionary
    start_p = {}
    for prev_tag, num_next_tags in trans_p.items():
        # Manually handling the initialization of keys
        start_p[prev_tag] = len(num_next_tags) / total_transitions
    for prev_tag, num_next_tags in trans_p.items():
        start_p[prev_tag] = len(num_next_tags) / total_transitions

    # Initialize the Viterbi table and path tracking
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        if obs[0] in output_p[y]:
            V[0][y] = start_p[y] * output_p[y][obs[0]]
        else:
            V[0][y] = start_p[y] * output_p[y].get('UNSEEN_TEST', 0)  # Use 0 if 'UNSEEN' not present
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            # Compute the maximum probability and the state which led to it
            max_prob = 0
            max_state = None
            if obs[t] in output_p[y]:

                for y0 in states:
                    trans_prob = trans_p[y0].get(y, 0)  # Use 0 if transition probability does not exist
                    prob = V[t-1][y0] * trans_prob * output_p[y][obs[t]]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = y0
            else:
                
                for y0 in states:
                    trans_prob = trans_p[y0].get(y, 0)  # Use 0 if transition probability does not exist
                    prob = V[t-1][y0] * trans_prob * output_p[y].get('UNSEEN_TEST', 0)
                    if prob > max_prob:
                        max_prob = prob
                        max_state = y0

            # Update the Viterbi table and paths
            V[t][y] = max_prob
            newpath[y] = path[max_state] + [y]

        # Don't need to remember the old paths
        path = newpath
        
    # Determine the best path for the final step
    n = 0  # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t

    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])

## Q3 (c) ### --------------------------------------------------------------------------------

# Viterbi prediction accuracy: 1024/1378 = 0.7431059506531205

## Q4 (a) ### --------------------------------------------------------------------------------

# The first step involves pre-processing by extracting suffixes from words, where applicable.
# For instance, transforming:
# 'testing' to 'ing'
# 'tested' to 'ed'
# Nouns and plural nouns are left unchanged, like:
# 'test' remains 'test'
# 'tests' remains 'tests'

# This approach stems from the belief that part-of-speech tagging largely depends on a word's tense or grammatical form.
# Extracting suffixes simplifies the tagging process by reducing the diversity of tokens the function needs to learn.
# For example, the common suffix 'ing' in 'testing' and 'dancing' helps in assigning an appropriate tag more easily, as it reduces the number of unique tokens to be processed.


# Stemming reduces words to their base or root form, a common technique in NLP for standardizing words by trimming suffixes.
# Stemming algorithms typically follow a set of rules for this reduction. It's intended to boost model accuracy by treating similar words identically.
# Since stemming is the opposite of what we require, we've devised a function to retrieve the part that was removed during stemming.
# For example, stemming 'testing' yields 'test', so our function aims to recover the removed part, 'ing'.


def extract_suffix(word: str) -> str:
    """
    Extract the suffix from a given word. 
    This is a simplified version of a suffix extraction process.
    """
    # Common suffixes in the English language
    common_suffixes = ['ing', 'ed', 's', 'er', 'tion', 'ly', 'ion', 'est', 'es' , 'ment']

    # Check if the word ends with a known suffix
    for suffix in common_suffixes:
        if word.endswith(suffix):
            # Return the suffix
            return suffix
    
    if word.startswith("@user_"):
        return "@user_"

    # If no known suffix is found, return the original word
    return word

# Test cases
# Uncomment these to test the function
# print(extract_suffix("testing"))  # Expected 'ing'
# print(extract_suffix("tested"))   # Expected 'ed'
# print(extract_suffix("test"))     # Expected 'test'
# print(extract_suffix("tests"))    # Expected 's'
# print(extract_suffix("playing"))  # Expected 'ing'
# print(extract_suffix("played"))   # Expected 'ed'
# print(extract_suffix("plays"))    # Expected 's'
# print(extract_suffix("play"))     # Expected 'play'

# In our next enhancement, we're focusing on handling unknown words more effectively through a rule-based system. This system utilizes word patterns to identify unseen words.
# For example, words starting with "@USER_" are most likely to be associated with the "@" tag.
# So, in cases where the current word starts with "@USER_" and the expected tag is "@", we assign a probability of 1 to this pairing, rather than relying on the standard probability for unknown words.
# Our rule-based system includes functions to recognize URLs, Hashtags, User mentions, Numbers, Retweets, and Punctuations, as outlined in question 5b.

# The final improvement aims at enhancing the modeling of transition probabilities by adopting second-order Hidden Markov Models (HMMs).
# In this approach, the probability of a tag depends not just on the preceding tag, but on the two preceding tags.
# For instance, in the tweet "@ A N .", we calculate the transition probability as P(N|A, @) rather than just P(N|A) as in a first-order HMM.
# By considering two preceding tags, second-order HMMs can more accurately capture the complexities in sentence structures, potentially improving prediction accuracy.

def compute_trigram_mle_probs(training_data):
    # Smoothing factor for calculating probabilities
    smoothing_factor = 0.1

    # Open the training data file and read the content
    training_file = open(training_data, "r", encoding='utf8')
    training_lines = [line for line in training_file.read().split('\n\n') if line]

    # Separate data into tag/token pairs for each sentence
    # Here, each tweet is treated as a sentence
    sentence_pairs = [line.split('\n') for line in training_lines]
    sentence_pairs = [[pair.split('\t') for pair in sentence] for sentence in sentence_pairs]

    # Process each token to extract suffix and convert to lowercase
    for sentence in sentence_pairs:
        sentence[:] = [(extract_suffix(word.lower()), tag) for word, tag in sentence]

    # Extract only the tags from each sentence
    tag_sequences = [[pair[1] for pair in sentence] for sentence in sentence_pairs]

    # Extract all tokens and tags using list comprehensions
    all_tokens = [pair[0] for sentence in sentence_pairs for pair in sentence]
    all_tags = [pair[1] for sentence in sentence_pairs for pair in sentence]

    # Identify unique tokens and tags from the dataset
    unique_tokens = list(set(all_tokens))
    unique_tags = list(set(all_tags))

    # Pair each tag with its corresponding token
    tag_token_pairs = list(zip(all_tags, all_tokens))

    # Initialize a dictionary to count transitions between tag triples
    transition_counts = {"*": {from_tag: 0 for from_tag in unique_tags}}
    for from_tag in unique_tags:
        transition_counts[from_tag] = {to_tag: 0 for to_tag in unique_tags}

    # Setting up nested structure for trigram probabilities
    transition_counts["*"]["*"] = {}
    for from_tag in transition_counts:
        for middle_tag in transition_counts:
            transition_counts[from_tag][middle_tag] = {to_tag: 0 for to_tag in transition_counts}
            transition_counts[from_tag][middle_tag]['END'] = 0
    transition_counts["*"]["*"]['*'] = 0

    # Populate the transition counts based on the tag sequences in the dataset
    for tweet_tags in tag_sequences:
        n = len(tweet_tags)
        if n > 1:
            transition_counts['*']['*'][tweet_tags[0]] += 1
        else:
            transition_counts['*'][tweet_tags[0]]['END'] += 1

        for i in range(1, n):
            if i == 1:
                transition_counts['*'][tweet_tags[i-1]][tweet_tags[i]] += 1
            else:
                transition_counts[tweet_tags[i-2]][tweet_tags[i-1]][tweet_tags[i]] += 1

        if n > 1:
            transition_counts[tweet_tags[n-2]][tweet_tags[n-1]]['END'] += 1

    # Convert the counts to probabilities with smoothing
    transition_probs = {}
    for from_tag in transition_counts:
        transition_probs[from_tag] = {}
        for middle_tag in transition_counts[from_tag]:
            total_transitions = sum(transition_counts[from_tag][middle_tag].values())
            denominator = total_transitions + smoothing_factor * (len(unique_tags) + 1)
            transition_probs[from_tag][middle_tag] = {
                to_tag: (count + smoothing_factor) / denominator 
                for to_tag, count in transition_counts[from_tag][middle_tag].items()
            }

    # Write transition probabilities to a file
    with open('trans_probs2.txt', 'w') as trans_probs_file:
        json.dump(transition_probs, trans_probs_file)

    # Initialize emission counts as a nested dictionary
    emission_counts = {}
    for tag in unique_tags:
        emission_counts[tag] = {token: 0 for token in unique_tokens + ['unknown_token']}

    # Populate emission counts based on the tag-token pairs
    for tag, token in tag_token_pairs:
        emission_counts[tag][token] += 1

    # Create emission probability dictionary with smoothing
    emission_probs = {}
    for tag in emission_counts:
        total_emissions = sum(emission_counts[tag].values())
        emission_denominator = total_emissions + smoothing_factor * (len(unique_tokens) + 1)
        emission_probs[tag] = {
            token: (count + smoothing_factor) / emission_denominator 
            for token, count in emission_counts[tag].items()
        }

    with open('output_probs2.txt', 'w') as output_probs_file:
        json.dump(emission_probs, output_probs_file)


## Q4 (b) ### --------------------------------------------------------------------------------    

# Utilizing word patterns to effectively manage words that haven't been seen before

# Check if a token is a username mention on social media
def check_user_mention(token):
    return token.startswith("@USER_")

 # Check if token starts with 'http://' or 'https://'
def check_url(token: str) -> bool:
    if token.startswith("http://") or token.startswith("https://"):
        # Check if the remainder of the token contains valid URL characters
        url_part = token.split("://")[1]
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=")
        return all(char in valid_chars for char in url_part)
    return False

# Check for decimals / fractions
def check_number(token: str) -> bool:
    if '.' in token:
        parts = token.split('.')
        return len(parts) == 2 and all(part.isdigit() for part in parts)
    for symbol in ['/', ',', '\\']:
        if symbol in token:
            parts = token.split(symbol)
            return len(parts) == 2 and all(part.isdigit() for part in parts)
    # Check if the token is an integer
    return token.isdigit()

# Check if a token is a punctuation mark
def check_punctuation(token: str) -> bool:
    punctuation_marks = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    return token in punctuation_marks

# Check if a token indicates a retweet
def check_retweet(token):
    return token == "RT"

# Check if a token starts with a hashtag
def check_hashtag(token):
    return token.startswith("#")

def calculate_token_probability(k, token, tag1, tag2, tag3, path, trans_probs, output_probs):
    # Check if token is in known outputs
    if token in output_probs[tag1]:
        return path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1] * output_probs[tag1][token]

    # Calculate base probability
    base_prob = path[k - 1][tag2][tag3] * trans_probs[tag3][tag2][tag1]

    # Check for special token types
    if check_hashtag(token) and tag1 == "#":
        return base_prob
    elif check_url(token) and tag1 == "U":
        return base_prob
    elif check_user_mention(token) and tag1 == "@":
        return base_prob
    elif check_retweet(token) and tag1 == "~":
        return base_prob
    elif check_number(token) and tag1 == "$":
        return base_prob
    elif check_punctuation(token) and tag1 == ",":
        return base_prob
    else:
        # Handle unknown tokens
        return base_prob * output_probs[tag1]['unknown_token']

    
def find_max_probability_and_tags(path, trans_probs, allTags, last):
    max_prob, max_backpointer = 0, (0, 0)
    for tag1 in allTags:
        if tag1 == "*":
            continue
        for tag2 in allTags:
            prob = path[last][tag1][tag2] * trans_probs[tag2][tag1]["END"]
            if prob > max_prob:
                max_prob, max_backpointer = prob, (tag2, tag1)
    return max_prob, max_backpointer

def backtrack_tag_sequence(backpointers, best_pair, total_length):
    # Initialize the list to store the sequence of tags
    tag_sequence = []

    # Extract the most probable last two tags from best_pair
    prev_tag1, prev_tag2 = best_pair

    # For tweets with a single word, add only the second tag
    if total_length == 1:
        tag_sequence.append(prev_tag2)
    else:
        # For longer tweets, add both tags and backtrack for the rest
        tag_sequence.append(prev_tag2)
        tag_sequence.append(prev_tag1)

        # Backtrack through the backpointer array
        for index in range(total_length - 2, 0, -1):
            current_tag = backpointers[index + 2][prev_tag2][prev_tag1]
            tag_sequence.append(current_tag)
            prev_tag2, prev_tag1 = prev_tag1, current_tag  # Update tags for next iteration

    # Reverse the sequence to get the correct order of tags
    tag_sequence.reverse()
    return tag_sequence

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):

    # load the full set of tags
    with open(in_tags_filename, "r", encoding = "utf8") as tagfile:
        allTags = tagfile.read().split()

    #load the transition probs dictionary
    with open(in_trans_probs_filename) as transfile:
        trans_probs = json.load(transfile)

    # load the output probs dictionary
    with open(in_output_probs_filename) as outputProbsFile:
        output_probs = json.load(outputProbsFile)

    # Load the file being tested, split into tweets, convert all tokens to lowercase and apply extract_suffix
    with open(in_test_filename, "r", encoding='utf8') as test:
        lines = test.read().split('\n\n')
        observations = [tweet.split('\n') for tweet in lines if tweet]
        observations = [[extract_suffix(token.lower()) for token in tweet] for tweet in observations]

    # Initialize results list and add the start symbol to the tags
    result = []
    allTags.append("*")

    # Iterate over each tweet in observations
    for tweet in observations:
        # Initialize dictionaries for probabilities and backpointers
        path, BP = {}, {}
        # Base case for Viterbi trigram model
        path[0] = {tag1: {tag2: 1 if tag1 == "*" and tag2 == "*" else 0 for tag2 in allTags} for tag1 in allTags}
        
        #Recursive case for Viterbi Trigram model
        for k, token in enumerate(tweet, start=1):
            path[k], BP[k] = {}, {}
            for tag1 in allTags:
                path[k][tag1], BP[k][tag1] = {}, {}
                for tag2 in allTags:
                    path[k][tag1][tag2], BP[k][tag1][tag2] = 0, 0
                    if tag1 == "*":
                        continue
                    # Integration of the function into your existing code
                    for tag3 in allTags:
                        prob = calculate_token_probability(k, token, tag1, tag2, tag3, path, trans_probs, output_probs)

                        # Update the path and backpointer if a higher probability is found
                        if prob > path[k][tag1][tag2]:
                            path[k][tag1][tag2] = prob
                            BP[k][tag1][tag2] = tag3

        max_prob, max_backpointer = 0, (0, 0)
        last = len(tweet)
        last = len(tweet)
        max_prob, max_backpointer = find_max_probability_and_tags(path, trans_probs, allTags, last)
        
        tweet_length = len(tweet)  # Length of the tweet
        tags_sequence = backtrack_tag_sequence(BP, max_backpointer, tweet_length)

        for tags in tags_sequence:
            result.append(tags)
        result.append("")

    with open(out_predictions_filename, 'w') as output:
        for tag in result:
            if tag == "":
                output.write('\n')
            else:
                output.write(tag + '\n')

## Q4 (c) ### --------------------------------------------------------------------------------    

# Viterbi2 prediction accuracy:  1086/1378 = 0.7880986937590712

##### --------------------------------------------------------------------------------    

def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)


def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir =  "C:/Users/Wade/OneDrive - National University of Singapore/Files/Y4S2/BT3102, Only Final Exam 27 Apr 1pm/Project/project_files hr copy" #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'

    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')
 
    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    

if __name__ == '__main__':
    ddir =  "C:/Users/Wade/OneDrive - National University of Singapore/Files/Y4S2/BT3102, Only Final Exam 27 Apr 1pm/Project/project_files hr copy" #your working dir
    in_train_filename = f'{ddir}/twitter_train.txt'
    naive_ouput(in_train_filename, "naive_output_probs.txt", delta=0.1)
    compute_mle_probs(in_train_filename, "output_probs.txt", "trans_probs.txt", delta=0.1)
    compute_trigram_mle_probs(in_train_filename)
    run()
