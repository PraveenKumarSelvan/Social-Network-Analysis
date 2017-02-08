# coding: utf-8.

from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pprint

pp = pprint.PrettyPrinter(indent=4)
common_friends_candi = Counter()
label_names={}

consumer_key = '9Mql0ocSNyUOVBIJhvB1rCOuM'
consumer_secret = '0O9RHbs0axtUlowpJxRWhXb8DSGMHK6fx64qB94EeBKcAoaSKf'
access_token = '951933259-5VKBlaCyUwLUCyKCW0bipR7k7bT1DxXkTBwF3rxO'
access_token_secret = 'v6jK6OUHfSRoCK0Qo9HcXTV4cuNe6ZFGtQQkcIGscno6F'

# This method is done for you. Make sure to put your credentials in the file twitter.cfg.
def get_twitter():
    """ 
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Reads a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Doctest to check if implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    names=[line.strip() for line in open(filename)]
    return names


# Handle's Twitter's rate limiting.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    
    getusers=[]
    output=robust_request(twitter,"users/lookup",{'screen_name':screen_names})
    getusers=output.json()
    return getusers
    
def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    friends=[]
    request = robust_request(twitter,'friends/ids',{'screen_name':screen_name , 'count': 5000})
    for r in request:
        friends.append(r)
    return sorted(friends, key=int)   

def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    
    user_details=[]
    for user in users:
        user_friends_map={}
        user_friends_map['screen_name'] = user.get('screen_name')
        user_friends_map['friends'] = get_friends(twitter,user.get('screen_name'))
        user_friends_map['friends_count'] = user.get('friends_count') 
        user_details.append(user_friends_map)
    return user_details

def print_num_friends(users):
    """Prints the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    for user in users:
        screen_name=user.get('screen_name')
        count=user.get('friends_count')
        print ("\n"+screen_name+" "+"%d" %count)
    exit

def count_friends(users):
    global common_friends_candi
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    
    
    for user in users:
        common_friends_candi= common_friends_candi + Counter(user['friends'])
    return(common_friends_candi)
    
    
def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    common_friends= []

    for i in range(len(users)):
        user1 = users[i]
        for k in range(i+1,len(users)):
            user2 = users[k]
            common=len(set(user1['friends']) & set(user2['friends']))
            common_friends.append((user1['screen_name'].encode("ascii") ,user2['screen_name'].encode("ascii") ,common))

    return common_friends

def followed_by_hillary_and_donald(users, twitter):
    """
    Return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    hilary_friends =[]
    donald_friends =[]
    for user in users:
        if (user['screen_name'] =='HillaryClinton'):
            hilary_friends = user['friends']
        elif (user['screen_name'] =='realDonaldTrump'):
            donald_friends = user['friends']

        common_friends = set(hilary_friends) & set(donald_friends)
        
    output=robust_request(twitter,"users/lookup",{'user_id':common_friends.pop()})
    output = output.json()
    return (output[0].get('screen_name'))

def create_graph(users, friend_counts):
    
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    ###TODO
    pass
    global label_names
    # Create a graph
    graph = nx.DiGraph()
    %matplotlib inline
    
    # Add a node
    for user in users:
        graph.add_node(user['screen_name'],label=user['screen_name'],node_size=1000,node_color='r')
        label_names[user['screen_name']]=user['screen_name']
        for follower in user['friends']:
            if(common_friends_candi[follower] > 1):
                graph.add_edge(user['screen_name'], follower,node_size=500,node_color='b')
    
    #nx.draw(graph, with_labels=True)
    return graph

def draw_network(graph, users, filename):
    """
    Draw the network to a file. Label the candidate nodes; the friend
    nodes has no labels (to reduce clutter).

    Methods used include networkx.draw_networkx, plt.figure, and plt.savefig.
    """
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(111)
    ax.set_title('Assignment-1', fontsize=12)
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos,with_label=True, font_size=12)
    

    #plt.tight_layout()
    plt.savefig(filename, format="PNG")
    plt.show()


def main():
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    users=add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')
    
if __name__ == '__main__':
    main()