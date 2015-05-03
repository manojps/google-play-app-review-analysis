from lxml import html
import requests
import psycopg2

app = open('app_id_list.txt','r')
app_link = app.readlines()
total_apps = app_link.__len__()

conn = psycopg2.connect("dbname='gplay' user='postgres' host='localhost' password=''")

j=0
while j < total_apps:
    url = app_link[j].strip()
    print url
    page = requests.get(url)
    tree = html.fromstring(page.text)
    review_date = tree.xpath('//*[@class="review-date"]/text()')
    review_text = tree.xpath('//*[@class="review-body"]//text()[position()=3]')
    review_rating = tree.xpath('//@aria-label')


    print 'review_date: ', review_date.__len__()
    print 'review_text: ', review_text.__len__()

    
    i=0
    
    while i < review_text.__len__():
        if review_text[i] != ' ':
            cur = conn.cursor()
            cur.execute("insert into review (review_date, review_text, rating, app_link) values (%s, %s, %s, %s)", (review_date[i], review_text[i], review_rating[i], app_link[j].strip()))
            conn.commit()
        i += 1
    j += 1

conn.close()
