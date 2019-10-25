import re
import json
import lxml.html
from urllib import robotparser
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
from throttle import Throttle
import string
from nltk.tokenize import word_tokenize
import numpy as np


def download(url, user_agent='wswp', num_retries=2, proxies=None):
    """ Download a given URL and return the page content
        args:
            url (str): URL
        kwargs:
            user_agent (str): user agent (default: wswp)
            proxies (dict): proxy dict w/ keys 'http' and 'https', values
                            are strs (i.e. 'http(s)://IP') (default: None)
            num_retries (int): # of retries if a 5xx error is seen (default: 2)
    """
    #print('Downloading:', url)
    headers = {'User-Agent': user_agent}
    try:
        resp = requests.get(url, headers=headers, proxies=proxies)
        html = resp.text
        
        #print(resp.title())
        #print(html.title)

        if resp.status_code >= 400:
            print('Download error:', resp.status_code)
            print(url)
            html = None
            if num_retries and 500 <= resp.status_code < 600:
                # recursively retry 5xx HTTP errors
                return download(url, num_retries - 1)
    except requests.exceptions.RequestException as e:
        print('Download error:', e)
        html = None
    return html


def get_robots_parser(robots_url):
    " Return the robots parser object using the robots_url "
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    rp.read()
    return rp


def get_links(html):
    """ Return a list of links (using simple regex matching)
        from the html content """
    # a regular expression to extract all links from the webpage
    webpage_regex = re.compile("""<a[^>]+href=["'](.*?)["']""", re.IGNORECASE)
    # list of all links from the webpage
    return webpage_regex.findall(html)


def link_crawler(start_url, link_regex, robots_url=None, user_agent='wswp',
                 proxies=None, delay=0.0001, max_depth=999999, max_count = 100):
    """ Crawl from the given start URL following links matched by link_regex.
    In the current implementation, we do not actually scrape any information.

        args:
            start_url (str): web site to start crawl
            link_regex (str): regex to match for links
        kwargs:
            robots_url (str): url of the site's robots.txt
                              (default: start_url + /robots.txt)
            user_agent (str): user agent (default: wswp)
            proxies (dict): proxy dict w/ keys 'http' and 'https', values
                            are strs (i.e. 'http(s)://IP') (default: None)
            delay (int): seconds to throttle between requests
                         to one domain (default: 3)
            max_depth (int): maximum crawl depth (to avoid traps) (default: 4)
    """
    i = 0
    crawl_queue = [start_url]
    result = []
    # keep track which URL's have seen before
    seen = {}
    if not robots_url:
        robots_url = '{}/robots.txt'.format(start_url)
    rp = get_robots_parser(robots_url)
    throttle = Throttle(delay)
    while crawl_queue and i <= max_count:
        url = crawl_queue.pop()
        # check url passes robots.txt restrictions
        if rp.can_fetch(user_agent, url):
            depth = seen.get(url, 0)
            if depth == max_depth:
                print('Skipping %s due to depth' % url)
                continue
            if i > max_count:
                print('Skipping %s due to exceed limit count' % url)
                continue
            throttle.wait(url)
            html = download(url, user_agent=user_agent, proxies=proxies)
            if not html:
                continue
            i+=1
            #print(i)
            yield WikiItem(html, url)
            # TODO: add actual data scraping here
            # filter for links matching our regular expression
            for link in get_links(html):
                if re.match('#(a-z)*', link):
                    continue
                if re.match(link_regex, link):
                    abs_link2 = urljoin(start_url, 'A/')
                    abs_link = urljoin(abs_link2, link)
                    if abs_link not in seen and len(abs_link) < 200:
                        seen[abs_link] = depth + 1
                        crawl_queue.append(abs_link)
        else:
            print('Blocked by robots.txt:', url)

def WikiItem(html, url):
    item = {}
    item['url'] = url
    soup = BeautifulSoup(html,'lxml') 
    a = [s.extract() for s in soup(['script', 'style', 'noscript'])]
    item['content'] = soup.get_text().strip()
    
    # a = [s.extract() for s in soup(['script', 'style', 'noscript'])]
    # tree = lxml.html.fromstring(html)
    # item['content'] = tree.text_content().strip()

    return item


def WikiItem(html, url):
    item = {}
    item['url'] = url
    soup = BeautifulSoup(html, 'lxml')
    a = [s.extract() for s in soup(['script', 'style', 'noscript'])]
    item['content'] = soup.get_text().strip()

    # a = [s.extract() for s in soup(['script', 'style', 'noscript'])]
    # tree = lxml.html.fromstring(html)
    # item['content'] = tree.text_content().strip()

    return item


def open_file():
    file = open('items.json', 'w')
    return file

def close_file(dictItem, file):
    json.dump(dictItem, file, indent=4)
    file.close()

def process_file(item, file, dictItem):
    dictItem['item'].append(item)

content = link_crawler('http://10.192.72.157:8000/wikipedia_es_all_2017-01/?', '/*')
#print(len(content))
file = open_file()
#i = 0
dictItem = {}
dictItem['item'] = []

for x in content:
    process_file(x, file, dictItem)
    #i+=1
#print(i)

close_file(dictItem, file)

vocabulary = set([])
doc_names = set([])
tuples = set([])

with open('items.json') as file:
    data = json.load(file)
    for item in data['item']:
        #tokenizar
        tokens = word_tokenize(item['content'])
        #minuscula
        tokens = [t.lower() for t in tokens]
        words = [w for w in tokens if w.isalpha()]

        for w in words:
            vocabulary.add(w)
            tuples.append((w, item['url']))

        doc_names.add(item['url'])

def create_term_document_matrix(line_tuples, document_names, vocab):
    '''Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
    line_tuples: A list of tuples, containing the name of the document and
    a tokenized line from that document.
    document_names: A list of the document names
    vocab: A list of the tokens in the vocabulary

    Let n = len(document_names) and m = len(vocab).

    Returns:
    td_matrix: A mxn numpy array where the number of rows is the number of documents
        and each column corresponds to a token in the corpus. A_ij contains the
        frequency with which word i occurs in document j.
    vocab: A list containing the tokens being represented by each column.
    '''
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(vocab))))

    matrix = np.zeros([len(vocab), len(document_names)])

    for document, tokens in line_tuples:
        column_id = docname_to_id.get(document, None)
        if column_id is None:
            continue
        for word in tokens:
            row_id = vocab_to_id.get(word, None)
            if row_id is None:
                continue
        matrix[row_id, column_id] += 1

    return matrix

def compute_cosine_similarity(vector1, vector2):
    num = np.dot(vector1, vector2)
    den1 = np.sqrt((vector1 ** 2).sum())
    den2 = np.sqrt((vector2 ** 2).sum())
    return num / (den1 * den2)


def rank_plays(target_play_index, term_document_matrix, similarity_fn):
    # BEGIN SOLUTION
    m, n = term_document_matrix.shape
    sims = np.zeros(n)
    v_tgt = get_column_vector(term_document_matrix, target_play_index)
    for i in range(n):
        v_doc = get_column_vector(term_document_matrix, i)
        sims[i] = similarity_fn(v_tgt, v_doc)
    sims_sort = np.argsort(-sims)
    return sims_sort
    # END SOLUTION

td_matrix = create_term_document_matrix(tuples, doc_names, vocabulary)

#ranks = rank_plays(doc_names[0], td_matrix, compute_cosine_similarity)



