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

class uefa_site_exporter:
    def __init__(self, tournament):
        self.tournament = tournament
        self.tourn_url = tournament.lower().replace(' ', '')
        self.con = HTTPConnection('www.uefa.com')

    def add_group_events(self, cal):
        for stage in xrange(1, 7):
            self.add_stage_events(cal, stage, None)

    def add_stage_events(self, cal, stage, session):
        strstage = str(stage) + num_ends.get(stage, 'th')

        url = '/{0}/season=2016/matches/day={1}'.format(self.tourn_url, stage)
        if not session is None:
            url += '/session={0}'.format(session)

        self.con.request('GET', url + '/index.html')
        res = self.con.getresponse()
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
            event['summary'] = trophy_symbol + ' {0} {1} round'.format(self.tournament, strstage)
            event['summary'] = trophy_symbol + ' {0} {1} round, Group '.format(self.tournament, strstage)\
                             + ','.join(sorted(groups))
            event['description'] = '\n'.join(full_summary)
            event.add('dtstart', d)
            cal.add_component(event)

def add_playoff_event(cal, tourn, rnd, rnd_full, sleg, sses, d):
    event = ical.Event()
    event['uid'] = str(uuid.uuid3(uuid.NAMESPACE_OID, str(d)))
    event['location'] = 'Europe'
    event['description'] = '{0}, {1} leg, {2} session'.format(rnd_full, sleg, sses)
    event['description'] = '{0}, {1} leg'.format(rnd_full, sleg)
    event.add('dtstart', d)
    cal.add_component(event)

# tournament = 'UEFA Champions League'
# tournament = 'UEFA Europa League'

def export_calendar(filename, tournament):
    cal = ical.Calendar()

    exporter = uefa_site_exporter(tournament)
    exporter.add_group_events(cal)

    # No information about matches on site until draw
    if tournament == 'UEFA Champions League':
        schedule = [{'round' : 'R16',
                     'round_full' : 'Round of 16',
                     'legs' : [[(16, 2), (17, 2), (23, 2), (24, 2)],
                                [(8, 3), (9, 3), (15, 3), (16, 3)]]},
                    {'round' : 'QF',
                     'round_full' : 'Quarter-final',
                     'legs' : [[(5, 4), (6, 4)], [(12, 4), (13, 4)]]},
                    {'round' : 'Semi-final',
                     'round_full' : 'Semi-final',
                     'legs' : [[(26, 4), (27, 4)], [(3, 5), (4, 5)]]}]
    elif tournament == 'UEFA Europa League':
        schedule = [{'round' : 'R32',
                     'round_full' : 'Round of 32',
                     'legs' : [[(18, 2)], [(25, 3)]]},
                    {'round' : 'R16',
                     'round_full' : 'Round of 16',
                     'legs' : [[(10, 3)], [(17, 3)]]},
                    {'round' : 'QF',
                     'round_full' : 'Quarter-final',
                     'legs' : [[(7, 4)], [(14, 4)]]},
                    {'round' : 'Semi-final',
                     'round_full' : 'Semi-final',
                     'legs' : [[(28, 4)], [(5, 5)]]}]

    for rnd in schedule:
        name = rnd['round']
        full = rnd['round_full']
        for nleg, leg in enumerate(rnd['legs'], 1):
            sleg = str(nleg) + num_ends.get(nleg, 'th')
            for ses, dt in enumerate(leg, 1):
                sses = str(ses) + num_ends.get(ses, 'th')
                add_playoff_event(cal, tournament, name, full, sleg, sses, date(2016, dt[1], dt[0]))

    with open(filename, 'wb') as f:
        f.write(cal.to_ical())

def update_score(filename, tournament, stage, session):
    cal = ical.Calendar()

    exporter = uefa_site_exporter(tournament)
    exporter.add_stage_events(cal, stage, session)

    with open(filename, 'wb') as f:
        f.write(cal.to_ical())

# Exmaple:
#  export_calendar(r'Q:\europa_league.ics', 'UEFA Europa League')
#  update_score(r'Q:\europa_league_5.ics', 'UEFA Europa League', 5, 1)
