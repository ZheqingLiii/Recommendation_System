# COEN 169 Project2
# Zheqing Li

import time
import numpy
import csv
import math
import sys as Sys

# for custom algorithm to find the median value in a list
def median(lst):
    return numpy.median(numpy.array(lst))



train = [[0 for x in range(1000)] for y in range(200)]
userRating = [] # for test file
predictRating = []

class Algorithms:
    cosine_sim, pearson, pearson_iuf, pearson_case, item_cos, custom1, custom2 = range(7)



def cosine_sim(user_id = None, movie_id = None):
#    print user_id
#    print movie_id
    k = 50
    k_w = []
    k_wr = []
    user_movieID = []
    user_rating = []
    rating = 0 # predicted rating

    for rec in userRating: # rec:[User, Movie, Rating]
        if rec[0] == user_id and rec[2] != 0:
            user_movieID.append(rec[1]) #catch rated movie id for current user
            #print user_movieID
            user_rating.append(rec[2]) # catch rated movie rating for current user
            #print user_rating

    # find avg rating of this user
    avg = 0.0
    for rating in user_rating:
        avg += rating
    if len(user_rating) != 0:
        avg = float(avg)/(len(user_rating))
    else:
        avg = 3.0

    for i, user in enumerate(train): # i is user index 0-199, user is list of 1000 movie ratings
        #calculate cosine sim
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        for i in range(0, len(user_movieID)):
            movieID = user_movieID[i]
            #print movieID
            if user[movieID-1] != 0:
                num += user[movieID-1] * user_rating[i]
                de1 += (user_rating[i])**2
                de2 += (user[movieID-1])**2
        de = float(math.sqrt(de1) * math.sqrt(de2))
        if de == 0:
            w_au = 0
        elif de != 0:
            w_au = float(num)/float(de)

        wr = 0.0
        if (w_au != 0) and (user[movie_id-1] != 0):   # user[movie_id] is r_ui
            wr = w_au * user[movie_id-1]

        if (w_au != 0) and (wr != 0):
            if len(k_w) < k:
                k_w.append(w_au)
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_w)
                index = -1
                for i, user in enumerate(k_w):
                    if user == smallest:
                        index = i
                if smallest < w_au:
                    k_w[i] = w_au
                    k_wr[i] = wr

    if sum(k_w) == 0:
        rating = 0
    else:
        rating = (float(sum(k_wr))) / (float(sum(k_w)))
    rating = round(rating)
    print rating

    if rating > 5:
        rating = 5
    if rating <= 0:
        rating = avg
    del k_w[:]
    del k_wr[:]
    del user_rating[:]
    del user_movieID[:]
    
    return int(rating)







def pearson(user_id = None, movie_id = None):
    #    print user_id
    #    print movie_id
    k = 50
    k_w = []
    k_absw = []
    k_wr = []
    user_movieID = []
    user_rating = []
    rating = 0.0 # predicted rating
    
    for rec in userRating: # rec:[User, Movie, Rating]
        if rec[0] == user_id and rec[2] != 0:
            user_movieID.append(rec[1]) #catch rated movie id for current user
            #print user_movieID
            user_rating.append(rec[2]) # catch rated movie rating for current user
            #print user_rating

    # find avg rating of this user
    avg = 0.0
    for rating in user_rating:
        avg += rating
    if len(user_rating) != 0:
        avg = float(avg)/(len(user_rating))
    else:
        avg = 3.0
    
    for i, user in enumerate(train): # i is user index 0-199, user is list of 1000 movie ratings
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        
        # find average rating of this user
        cur_avg = 0.0
        cur_num = 0
        for movie in range(0, 999):
            if user[movie] != 0:
                cur_avg += user[movie]
                cur_num += 1
        cur_avg = float(float(cur_avg)/cur_num)
        
        for i in range(0, len(user_movieID)):
            movieID = user_movieID[i]
            #print movieID
            if user[movieID-1] != 0:
                num += float(user[movieID-1] - cur_avg) * float(user_rating[i] - avg)
                de1 += float(user_rating[i] - avg)**2
                de2 += float(user[movieID-1] - cur_avg)**2
        de = float(math.sqrt(de1) * math.sqrt(de2))
        if de == 0 or num == 0:
            w_au = 0
        elif de != 0 and num != 0:
            w_au = float(num)/float(de)

        wr = 0.0
        if (w_au != 0) and (user[movie_id-1] != 0): # user[movie_id] is r_ui
            wr = float(w_au) * float(user[movie_id-1] - float(cur_avg))


        if w_au != 0 and wr != 0:
            if len(k_w) < k:
                k_w.append(w_au)
                k_absw.append(abs(w_au))
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_absw)
                index = -1
                for i, user in enumerate(k_absw):
                    if user == smallest:
                        index = i
                if smallest < abs(w_au):
                    k_w[i] = w_au
                    k_absw[i] = abs(w_au)
                    k_wr[i] = wr

    if sum(k_absw) == 0:
        rating = 0.0
    else:
        rating = float(avg) + float(float(sum(k_wr))) / (float(sum(k_absw)))
        rating = round(rating)
    
    if rating > 5:
        rating = 5
    if rating <= 0:
        rating = round(avg)
    del k_w[:]
    del k_absw[:]
    del k_wr[:]
    del user_rating[:]
    del user_movieID[:]

    print rating
    return int(rating)










def iuf(movieID):
    mj = 0
    m = 200
    fj = 0.0
    for i, user in enumerate(train):
        if user[movieID] != 0:
            mj += 1
    if mj != 0:
        fj = float(math.log(float(m/mj), 2))
    else:
        fj = 0.0
    return fj

def pearson_iuf(user_id = None, movie_id = None):
    #    print user_id
    #    print movie_id
    k = 50
    k_w = []
    k_absw = []
    k_wr = []
    user_movieID = []
    user_rating = []
    rating = 0 # predicted rating
    
    for rec in userRating: # rec:[User, Movie, Rating]
        if rec[0] == user_id and rec[2] != 0:
            user_movieID.append(rec[1]) #catch rated movie id for current user
            #print user_movieID
            user_rating.append(rec[2]) # catch rated movie rating for current user
            #print user_rating

    # find avg rating of this user
    avg = 0.0
    for rating in user_rating:
        avg += rating
    if len(user_rating) != 0:
        avg = float(avg)/(len(user_rating))
    else:
        avg = 3.0

    test_user_sum = 0.0
    test_user_sumsqr = 0.0
    for i in range(0, len(user_movieID)):
        rate = float(user_rating[i] - avg)
        test_user_sum += float(rate * float(iuf(user_movieID[i]-1)))
        test_user_sumsqr += float(rate * rate * float(iuf(user_movieID[i]-1)))

    for i, user in enumerate(train): # i is user index 0-199, user is list of 1000 movie ratings
        num1 = 0.0
        cur_user_sum = 0.0
        cur_user_sumsqr = 0.0
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        
        
        # find average rating of this user
        cur_avg = 0.0
        cur_num = 0
        for movie in range(0, 999):
            if user[movie] != 0:
                cur_avg += user[movie]
                cur_num += 1
        cur_avg = float(float(cur_avg)/cur_num)
        
        for i in range(0, len(user_movieID)):
            movieID = user_movieID[i]
            #print movieID
            if user[movieID-1] != 0:
                fj = float(iuf(movieID-1))
                rate = float(user[movieID-1] - cur_avg)
                cur_user_sum += float(rate * fj)
                cur_user_sumsqr += float(fj * rate * rate)
                num1 += float(float(user[movieID-1] - cur_avg) * float(user_rating[i] - avg) * float(fj))
                print num1
                
                de1 += float(test_user_sumsqr - float((test_user_sum)**2))
                print de1
                de2 += float(cur_user_sumsqr - float((cur_user_sum)**2))
                print de2
        num = float(num1 - test_user_sum * cur_user_sum)
        print num
        de = float(math.sqrt(abs(de1 * de2)))
        if de == 0 or num == 0:
            w_au = 0
        elif de != 0 and num != 0:
            w_au = float(num)/float(de)
        
        wr = 0.0
        if (w_au != 0) and (user[movie_id-1] != 0): # user[movie_id] is r_ui
            wr = float(w_au) * float(user[movie_id-1] - float(cur_avg))

        
        if w_au != 0 and wr != 0:
            if len(k_w) < k:
                k_w.append(w_au)
                k_absw.append(abs(w_au))
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_absw)
                index = -1
                for i, user in enumerate(k_absw):
                    if user == smallest:
                        index = i
                if smallest < abs(w_au):
                    k_w[i] = w_au
                    k_absw[i] = abs(w_au)
                    k_wr[i] = wr

    if sum(k_absw) == 0:
        rating = 0
    else:
        rating = float(avg) + float(float(sum(k_wr))) / (float(sum(k_absw)))
        rating = round(rating)

    if rating > 5:
        rating = 5
    if rating <= 0:
        rating = round(avg)
    del k_w[:]
    del k_absw[:]
    del k_wr[:]
    del user_rating[:]
    del user_movieID[:]
    
    print rating
    return int(rating)










def pearson_case(user_id = None, movie_id = None):
    #    print user_id
    #    print movie_id
    k = 50
    k_w = []
    k_absw = []
    k_wr = []
    user_movieID = []
    user_rating = []
    rating = 0 # predicted rating
    
    for rec in userRating: # rec:[User, Movie, Rating]
        if rec[0] == user_id and rec[2] != 0:
            user_movieID.append(rec[1]) #catch rated movie id for current user
            #print user_movieID
            user_rating.append(rec[2]) # catch rated movie rating for current user
    #print user_rating

    # find avg rating of this user
    avg = 0.0
    for rating in user_rating:
        avg += rating
    if len(user_rating) != 0:
        avg = float(avg)/(len(user_rating))
    else:
        avg = 3.0
    
    for i, user in enumerate(train): # i is user index 0-199, user is list of 1000 movie ratings
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        
        # find average rating of this user
        cur_avg = 0.0
        cur_num = 0
        for movie in range(0, 999):
            if user[movie] != 0:
                cur_avg += user[movie]
                cur_num += 1
        cur_avg = float(float(cur_avg)/cur_num)
        
        for i in range(0, len(user_movieID)):
            movieID = user_movieID[i]
            #print movieID
            if user[movieID-1] != 0:
                num += float(user[movieID-1] - cur_avg) * float(user_rating[i] - avg)
                de1 += float(user_rating[i] - avg)**2
                de2 += float(user[movieID-1] - cur_avg)**2
        de = float(math.sqrt(de1) * math.sqrt(de2))
        if de == 0 or num == 0:
            w_au = 0
        elif de != 0 and num != 0:
            w_au = float(num)/float(de)
            w_au = float(w_au) * float(math.pow(abs(w_au), 2.5-1))

        wr = 0.0
        if (w_au != 0) and (user[movie_id-1] != 0): # user[movie_id] is r_ui
            wr = float(w_au) * float(user[movie_id-1] - float(cur_avg))
        
        if w_au != 0 and wr != 0:
            if len(k_w) < k:
                k_w.append(w_au)
                k_absw.append(abs(w_au))
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_absw)
                index = -1
                for i, user in enumerate(k_absw):
                    if user == smallest:
                        index = i
                if smallest < abs(w_au):
                    k_w[i] = w_au
                    k_absw[i] = abs(w_au)
                    k_wr[i] = wr

    if sum(k_absw) == 0:
        rating = 0
    else:
        rating = float(avg) + float(float(sum(k_wr))) / (float(sum(k_absw)))
        rating = round(rating)

    if rating > 5:
        rating = 5
    if rating <= 0:
        rating = round(avg)
    del k_w[:]
    del k_absw[:]
    del k_wr[:]
    del user_rating[:]
    del user_movieID[:]
    
    print rating
    return int(rating)










# item based adjusted cosine similarity
def item_cos(user_id = None, movie_id = None):
    k = 100
    user_avg = []
    k_w = []
    k_absw = []
    k_wr = []
    user_ID = []
    user_rating = []
    rating = 0 # predicted rating
    
    # find all the user averages in train
    for user in train:
        num = 0
        average = 0.0
        for i in range(0, 999):
            if user[i] != 0:
                average += user[i]
                num += 1
        user_avg.append(float(average/num))
    print user_avg
    
    
    for index, user in enumerate(train):
        if user[movie_id-1] != 0:
            user_ID.append(index+1)
            user_rating.append(user[movie_id-1])

    avg = 0.0
    num = 0
    for rec in userRating:
        if rec[0] == user_id:
            if rec[2] != 0:
                avg += rec[2]
                num += 1
    avg = float(avg/num) # average of current test user

    for movieID in range(0, 999):
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        for userID in user_ID: # userID range 0~999
            if train[userID-1][movieID] != 0:
                num += float(train[userID-1][movieID] - user_avg[userID-1]) * float(train[userID-1][movie_id-1] - avg)
                de1 += float(float(train[userID-1][movieID] - user_avg[userID-1]))**2
                de2 += float(float(train[userID-1][movie_id-1] - avg))**2

        de = float(math.sqrt(float(de1 * de2)))
        if de == 0 or num == 0:
            w_au = 0.0
        else:
            w_au = float(num/de)
        print w_au

        #find r_ui in test user
        r = 0
        for rec in userRating:
            if rec[0] == user_id:
                if rec[1] == movieID+1:
                    r = rec[2]

        wr = 0.0
        if (w_au != 0) and (r != 0):
            r = float(r - avg)
            wr = float(w_au * r)


        if w_au != 0 and wr != 0:
            if len(k_w) < k:
                k_w.append(w_au)
                k_absw.append(abs(w_au))
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_absw)
                index = -1
                for i, user in enumerate(k_absw):
                    if user == smallest:
                        index = i
                if smallest < abs(w_au):
                    k_w[i] = w_au
                    k_absw[i] = abs(w_au)
                    k_wr[i] = wr

    if sum(k_absw) == 0:
        rating = 0
    else:
        rating = float(avg) + float(float(sum(k_wr))) / (float(sum(k_absw)))
        rating = round(rating)

    if rating > 5:
        rating = 5
    if rating <= 0:
        rating = round(avg)

    del k_w[:]
    del k_absw[:]
    del k_wr[:]
    del user_rating[:]
    del user_ID[:]
    del user_avg[:]

    print rating
    return int(rating)











def custom1(user_id = None, movie_id = None):
    
    k = 50
    k_w = []
    k_absw = []
    k_wr = []
    user_movieID = []
    user_rating = []
    rating = 0.0 # predicted rating
    
    for rec in userRating: # rec:[User, Movie, Rating]
        if rec[0] == user_id and rec[2] != 0:
            user_movieID.append(rec[1]) #catch rated movie id for current user
            #print user_movieID
            user_rating.append(rec[2]) # catch rated movie rating for current user
    #print user_rating

    # find median rating of this user
    med = float(median(user_rating))
    if med == 0:
        med = 3.0

    for i, user in enumerate(train): # i is user index 0-199, user is list of 1000 movie ratings
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        
        # find average rating of this user
        cur_rating = []
        cur_med = 0.0
        for movie in range(0, 999):
            if user[movie] != 0:
                cur_rating.append(user[movie])
        cur_med = float(median(cur_rating))
        if cur_med == 0.0:
            cur_med = 3.0

        del cur_rating[:]
    
        for i in range(0, len(user_movieID)):
            movieID = user_movieID[i]
            #print movieID
            if user[movieID-1] != 0:
                num += float(user[movieID-1] - cur_med) * float(user_rating[i] - med)
                de1 += float(user_rating[i] - med)**2
                de2 += float(user[movieID-1] - cur_med)**2
        de = float(math.sqrt(de1) * math.sqrt(de2))
        if de == 0 or num == 0:
            w_au = 0
        elif de != 0 and num != 0:
            w_au = float(num)/float(de)
        
        wr = 0.0
        if (w_au != 0) and (user[movie_id-1] != 0): # user[movie_id] is r_ui
            wr = float(w_au) * float(user[movie_id-1] - float(cur_med))
        
        
        if w_au != 0 and wr != 0:
            if len(k_w) < k:
                k_w.append(w_au)
                k_absw.append(abs(w_au))
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_absw)
                index = -1
                for i, user in enumerate(k_absw):
                    if user == smallest:
                        index = i
                if smallest < abs(w_au):
                    k_w[i] = w_au
                    k_absw[i] = abs(w_au)
                    k_wr[i] = wr

    if sum(k_absw) == 0:
        rating = 0.0
    else:
        rating = float(med) + float(float(sum(k_wr))) / (float(sum(k_absw)))
        rating = round(rating)

    if rating > 5:
        rating = 5
    if rating <= 0:
        rating = round(med)
    del k_w[:]
    del k_absw[:]
    del k_wr[:]
    del user_rating[:]
    del user_movieID[:]
    
    print rating
    return int(rating)














def custom2(user_id = None, movie_id = None):

    k = 35
    k_w = []
    k_wr = []
    user_movieID = []
    user_rating = []
    rating = 0 # predicted rating

    for rec in userRating: # rec:[User, Movie, Rating]
        if rec[0] == user_id and rec[2] != 0:
            user_movieID.append(rec[1]) #catch rated movie id for current user
            #print user_movieID
            user_rating.append(rec[2]) # catch rated movie rating for current user
            #print user_rating

    # find avg rating of this user
    avg = 0.0
    for rating in user_rating:
        avg += rating
    if len(user_rating) != 0:
        avg = float(avg)/(len(user_rating))
    else:
        rate = item_cos(user_id, movie_id)
        return rate

    for i, user in enumerate(train): # i is user index 0-199, user is list of 1000 movie ratings
        #calculate cosine sim
        num = 0.0
        de1 = 0.0
        de2 = 0.0
        for i in range(0, len(user_movieID)):
            movieID = user_movieID[i]
            #print movieID
            if user[movieID-1] != 0:
                num += user[movieID-1] * user_rating[i]
                de1 += (user_rating[i])**2
                de2 += (user[movieID-1])**2
        de = float(math.sqrt(de1) * math.sqrt(de2))
        if de == 0:
            w_au = 0
        elif de != 0:
            w_au = float(num)/float(de)

        wr = 0.0
        if (w_au != 0) and (user[movie_id-1] != 0):   # user[movie_id] is r_ui
            wr = w_au * user[movie_id-1]

        if (w_au != 0) and (wr != 0):
            if len(k_w) < k:
                k_w.append(w_au)
                k_wr.append(wr)
            else:
                # find smallest similarity in k users, and check if the current sim is larger
                smallest = min(k_w)
                index = -1
                for i, user in enumerate(k_w):
                    if user == smallest:
                        index = i
                if smallest < w_au:
                    k_w[i] = w_au
                    k_wr[i] = wr

    if sum(k_w) == 0:
        rate = item_cos(user_id, movie_id)
        return rate
    else:
        rating = (float(sum(k_wr))) / (float(sum(k_w)))
    rating = round(rating)
    print rating

    if rating > 5:
        rating = 5
    if rating <= 0:
        rate = item_cos(user_id, movie_id)
        return rate
    del k_w[:]
    del k_wr[:]
    del user_rating[:]
    del user_movieID[:]
    
    return int(rating)















# function for algorithm driver
def driver(alg = None):
    global predictRating
    global userRating
    start = time.time()
    for i, rec in enumerate(userRating):
        rating = 0
        if rec[2] == 0: # for unrated movies
            if alg == Algorithms.cosine_sim:
                rating = cosine_sim(rec[0], rec[1])
            elif alg == Algorithms.pearson:
                rating = pearson(rec[0], rec[1])
            elif alg == Algorithms.pearson_iuf:
                rating = pearson_iuf(rec[0], rec[1])
            elif alg == Algorithms.pearson_case:
                rating = pearson_case(rec[0], rec[1])
            elif alg == Algorithms.item_cos:
                rating = item_cos(rec[0], rec[1])
            elif alg == Algorithms.custom1:
                rating = custom1(rec[0], rec[1])
            elif alg == Algorithms.custom2:
                rating = custom2(rec[0], rec[1])
            
            predictRating.append(([rec[0]] + [rec[1]] + [rating]))




# functions for reading file
def read_file(file_to_read, deli):
    data = []
    with open(file_to_read, "rU") as input:
        reader = csv.reader(input, delimiter = deli)
        reader = reader_int(reader)
        data = list(reader)
    return data
def reader_int(reader):
    #cvs reader
    for v in reader:
        yield map(int, v)



# function for writing file
def write_file(output, data, deli):
    with open(output, "wb") as outFile:
        writer = csv.writer(outFile, delimiter = deli)
        for num in data:
            writer.writerow(num)



def main():
    # main function
    # read data from train.txt into train, and let user choose test files and algorithms
    global train
    global userRating

    train = read_file("train.txt", "\t")
    # test the train.txt has been read correctly
    #write_file("output.txt", train, " ")
    testSelect = 10
    algSelect = 10

    while testSelect != 0:
        testSelect = input( """Please choose a test file
            1: test5.txt   2: test10.txt   3: test20.txt   0: Quit\n> """)

        testFile = ""
        outFile = ""
        if testSelect == 1:
            testFile = "test5.txt"
            outFile = "result5.txt"
        elif testSelect == 2:
            testFile = "test10.txt"
            outFile = "result10.txt"
        elif testSelect == 3:
            testFile = "test20.txt"
            outFile = "result20.txt"
        elif testSelect == 0:
            break
        else:
            print "Please retype a valid input."
            continue


        del userRating[:]
        userRating = read_file(testFile, " ")
        # test testFiles has been read correctly
        #write_file("output.txt", userRating, " ")
        del predictRating[:]

        while algSelect != 0:
            print "Please select an algorithm for testing " + testFile
            algSelect = input( """
            1: Cosine Similarity
            2: Pearson Correlation
            3: Pearson Correlation - Inverse User Frequency
            4: Pearson Correlation - Case Amplification
            5: Item based adjusted cosine similarity
            6: Custom1
            7: Custom2
            0: Exit\n>""")

            if algSelect == 1:
                driver(Algorithms.cosine_sim)
            elif algSelect == 2:
                driver(Algorithms.pearson)
            elif algSelect == 3:
                driver(Algorithms.pearson_iuf)
            elif algSelect == 4:
                driver(Algorithms.pearson_case)
            elif algSelect == 5:
                driver(Algorithms.item_cos)
            elif algSelect == 6:
                driver(Algorithms.custom1)
            elif algSelect == 7:
                driver(Algorithms.custom2)
            elif algSelect == 0:
                break
            else:
                print "Please retype a valid input."
                continue

            write_file(outFile, predictRating, " ")
            print "\nDone. output file: " + outFile




if __name__ == '__main__':
    main()
