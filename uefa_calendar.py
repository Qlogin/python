# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:08:09 2015

@author: q-login
"""

import re
import uuid
import pytz
from datetime import date, time, datetime, timedelta

from htmldom import htmldom
from httplib import HTTPConnection

import icalendar as ical

num_ends = {1 : 'st', 2 : 'nd',  3 :'rd'}
ball_symbol   = '\xe2\x9a\xbd'
trophy_symbol = '\xf0\x9f\x8f\x86'

cal = ical.Calendar()
con = HTTPConnection('www.uefa.com')
for stage in xrange(1, 9):
    strstage = str(stage) + num_ends.get(stage, 'th')

    con.request('GET', '/uefachampionsleague/season=2016/matches/day={0}/index.html'.format(stage))
    res = con.getresponse()
    page = res.read()

    dom = htmldom.HtmlDom()
    dom = dom.createDom(page)
    root = dom.find('div#matchesindex')
    assert root.len == 1
    assert root.attr('class') == 'matchesbycalendar'
    tables = root.children('table')
    for i in xrange(0, tables.len):
        tbl = tables[i]
        dmatch = re.search('date(\d{4})(\d{2})(\d{2})', tbl.attr('class'))
        if dmatch is None:
            continue
        d = date(*map(int, dmatch.groups()))

        full_summary = []
        groups       = set([])

        tbody = tbl.find('tbody')
        for j in xrange(0, tbody.len):
            bd = tbody[j]
            if bd.attr('class') == 'promo-row':
                continue

            start = bd.find('tr.match_res a.sc')
            tmatch = re.search('(\d{2})\.(\d{2})', start.text())
            delim  = '-'
            if tmatch is None:
                t = time(20, 45, tzinfo=pytz.timezone('CET'))
                delim = start.text().replace('\n', '').replace('-', ':')
            else:
                t = time(*map(int, tmatch.groups()), tzinfo=pytz.timezone('CET'))
            dt = datetime.combine(d, t)

            stad = bd.find('span.stadium_name')
            stad_name = ''.join(map(lambda x: stad[x].text(), xrange(0, stad.len)))

            gname = bd.find('span.gname>a')
            group = gname.text()[-1]

            home = dict(full='', short='')
            away = dict(full='', short='')

            tds = bd.find('tr.match_res>td')
            for k in xrange(0, tds.len):
                classval = tds[k].attr('class')
                if classval == 'r home nob':
                    home['short'] = tds[k].find('a').text()
                elif classval == 'logo home-logo nob':
                    home['full'] = tds[k].find('a>img').attr('title')
                elif classval == 'l away nob':
                    away['short'] = tds[k].find('a').text()
                elif classval == 'logo away-logo nob':
                    away['full'] = tds[k].find('a>img').attr('title')

            summary = '{0} {4} {1} ({2} round, "{3}")'\
                        .format(home['short'], away['short'], strstage, group, delim)
            full_summary.append(summary)
            groups.add(group)

            event = ical.Event()
            event['uid'] = str(uuid.uuid3(uuid.NAMESPACE_OID, home['short'] + away['short'] + str(d)))
            event['location'] = stad_name
            event['summary']  = ball_symbol + ' ' + summary
            event['description'] = '{0} vs. {1} ({2} round, Group {3}) at {4}'\
                               .format(home['full'], away['full'], strstage, group, stad_name)
            event.add('dtstart', dt)
            event.add('dtend', dt + timedelta(hours=2))
            cal.add_component(event)

        event = ical.Event()
        event['uid'] = str(uuid.uuid3(uuid.NAMESPACE_OID, str(d)))
        event['location'] = 'Europe'
        event['summary']  = trophy_symbol + ' UEFA Champions League {0} round, Group '.format(strstage) + ','.join(sorted(groups))
        event['description'] = '\n'.join(full_summary)
        event.add('dtstart', d)
        cal.add_component(event)

with open('uefa.ics', 'wb') as f:
    f.write(cal.to_ical())