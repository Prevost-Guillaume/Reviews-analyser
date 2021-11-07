import requests
from requests_html import HTMLSession
import codecs
from scrapy.selector import Selector
import streamlit as st


def get_next_page(html, base_url):
    sel  = Selector(text=html)
    next_url = sel.xpath('//*[@id="component_15"]/div/div[3]/div[8]/div/a').extract()
    next_url = [i for i in next_url if 'Next' in i]
    if next_url == []:
        return None
    next_url = next_url[0].split('href=')
    next_url = next_url[1].split('"')[1]
    return base_url+next_url



def getCommentsOnPage(html):
    comments = []
    sel  = Selector(text=html)
    for i in range(5):
        comment = ''
        # Cas la plus fréquent
        xp = sel.xpath(f'//*[@id="component_15"]/div/div[3]/div[{i+3}]/div[2]/div[3]/div[1]/div[1]/q/span/text()').get()
        if xp is None:
            # S'il y a une image dans un commentaire, le XPath change
            xp = sel.xpath(f'//*[@id="component_15"]/div/div[3]/div[{i+3}]/div[3]/div[3]/div[1]/div[1]/q/span/text()').get()
        if xp != None:
            comment += xp

        # Second part of comment
        xp = sel.xpath(f'//*[@id="component_15"]/div/div[3]/div[{i+3}]/div[2]/div[3]/div[1]/div[1]/q/span[2]/text()').get()
        if xp is None:
            # S'il y a une image dans un commentaire, le XPath change
            xp = sel.xpath(f'//*[@id="component_15"]/div/div[3]/div[{i+3}]/div[3]/div[3]/div[1]/div[1]/q/span[2]/text()').get()
        if xp != None:
            comment += xp

        if comment != '':
            comments.append(comment)
    return comments
        

def scrape_tripadvisor(url, n=1e3):
    comments = []
    s  = HTMLSession()
    response = s.get(url)
    while len(comments)<n:
        comments += getCommentsOnPage(response.html.html)
        url = get_next_page(response.html.html, 'https://www.tripadvisor.in')
        if url == None:
            break
        print(len(comments))
        s  = HTMLSession()
        response = s.get(url)
    return comments




def scrape_tripadvisor_st(url, progress_bar, n=1e3):
    comments = []
    s  = HTMLSession()
    response = s.get(url)
    while len(comments)<n:
        comments += getCommentsOnPage(response.html.html)
        url = get_next_page(response.html.html, 'https://www.tripadvisor.in')
        if url == None:
            break
        print(len(comments))
        s  = HTMLSession()
        response = s.get(url)
        progress_bar.progress(len(comments)/n)
    return comments







if __name__ == '__main__':
    ref_produit = ""
    url = 'https://www.tripadvisor.in/Hotel_Review-g187147-d228694-Reviews-Hotel_Malte_Astotel-Paris_Ile_de_France.html'
    comments = scrape_tripadvisor(url, n=10)

    print(len(comments))
    for c in comments:
        try:
            print('\n',c)
        except:
            pass

