import requests
import pandas as pd
import os
import time
from sklearn.metrics import accuracy_score
from statistics import mean
from concurrent.futures import ThreadPoolExecutor
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def test(model_name):
    """
    Test the model with the test data
    Args:
        model_name: the name of the model
    """

    print("Testing model: " + model_name)
    strings = pd.read_csv('Data/X_test.csv')
    result = pd.read_csv('Data/y_test.csv')

    # Number of strings to test
    input_len = 1000

    # Get the data
    strings = strings.to_numpy()[:input_len]
    result = result.to_numpy()[:input_len]

    # Prepare the data labels
    precessed_result = [2 if value == 5 else 1 if value == 3 else 0 for value in result]

    prediction = []
    model_inference_time = []
    results = []

    # Create the endpoint
    url = 'http://18.217.161.250/app/' + model_name + '/'

    def post_request(args):
        """
        Send a post request to the endpoint
        Args:
            args[0]: the string to test
            args[1]: the model name
        """
        response = requests.post(args[0], data = args[1])
        return [response,  args[1]]

    # Create the urls
    urls = [(url, json.dumps({"review": str(string[0]), "result": str(result)})) for string, result in zip(strings, precessed_result)]

    start = time.time()
    count = 1
    failed = 1

    # Create the TrheadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        for response in executor.map(post_request, urls):
            try:
                response = response[0].json()
            except:
                print("\nserver is down: " + str(failed) + " Count: " + str(count))
                failed += 1
                #print(response)
                break
            print("Done - " + str(count) + ": " + str(input_len), end='\r')
            count += 1
            prediction.append(int(response["rating"]))
            model_inference_time.append(float(response["time"]))
            results.append(int(response["result"]))

    # Print the results
    print("\n")
    print("Time Elapsed: " + str(round(time.time() - start, 4)))
    print("Average responce time: " + str(round((time.time() - start)/input_len, 4)))
    print("Accuracy: " + str(accuracy_score(results, prediction)))
    print("Average model inference time:", round(mean(model_inference_time),4), "seconds")
    print("Total inference time:", round(sum(model_inference_time),4), "seconds")
    print("\n>>>>>>>>>>>>>>>>>  END  <<<<<<<<<<<<<<<<<\n")

if __name__ == "__main__":
    print("\n>>>>>>>>>>>>>>>>>  START  <<<<<<<<<<<<<<<<<\n")
    # test("bert12")
    # test("bert8")
    # test("bert6")
    # test("bert4")
    # test("bert2")
    # test("bertteacher")
    # test("baselinestudent")
    # test("distilledstudent")

    # For testing Loadbalancer results
    print("\nTesting with loadbalancer (for bert12)")
    start = time.time()
    test("bertteacher")
    print("Total time: " + str(round(time.time() - start, 4)))
