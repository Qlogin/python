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

countries = {'ALB' : 'Albania',
             'AUT' : 'Austria',
             'AZE' : 'Azerbaijan',
             'BEL' : 'Belgium',
             'BLR' : 'Belarus',
             'CYP' : 'Cyprus',
             'CZE' : 'Czech Republic',
             'DEN' : 'Denmark',
             'ENG' : 'England',
             'ESP' : 'Spain',
             'FRA' : 'France',
             'GER' : 'Germany',
             'GRE' : 'Greece',
             'ITA' : 'Italy',
             'NED' : 'Netherlands',
             'NOR' : 'Norway',
             'POL' : 'Poland',
             'POR' : 'Portugal',
             'RUS' : 'Russia',
             'SCO' : 'Scotland',
             'SRB' : 'Serbia',
             'SUI' : 'Switzerland',
             'TUR' : 'Turkey',
             'UKR' : 'Ukraine'}

num_ends = {1 : 'st', 2 : 'nd',  3 :'rd'}
ball_symbol   = '\xe2\x9a\xbd'
trophy_symbol = '\xf0\x9f\x8f\x86'

class uefa_site_exporter:
    CHAMPIONS_LEAGUE = 'UEFA Champions League'
    EUROPA_LEAGUE = 'UEFA Europa League'

    def __init__(self, tournament):
        self.tournament = tournament
        self.tourn_url = tournament.lower().replace(' ', '')
        self.con = HTTPConnection('www.uefa.com')

    def add_group_events(self, cal):
        for stage in xrange(1, 7):
            self.add_stage_events(cal, stage, None)

    def add_playoff_events(self, cal):
        for stage in xrange(7, 9):
            self.add_stage_events(cal, stage, None)

    def add_stage_events(self, cal, stage, session):
        strstage = str(stage) + num_ends.get(stage, 'th')

        url = '/{0}/season=2016/matches/day={1}/index.html'.format(self.tourn_url, stage)
        tbl_id = 'md_' + str(stage)
        if not session is None:
            tbl_id += '_' + str(session)

        self.con.request('GET', url)
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
            if tbl.attr('id').find(tbl_id) == -1:
                continue

            dmatch = re.search('date(\d{4})(\d{2})(\d{2})', tbl.attr('class'))
            if dmatch is None:
                continue
            d = date(*map(int, dmatch.groups()))

            full_summary = []
            groups = set([])
            sround = ''
            leg = 2 - (stage % 2)
            sleg = str(leg) + num_ends[leg]

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
                ccode = re.search(r' \(([A-Z]{3})\)', stad_name)
                if ccode:
                    stad_name = stad_name.replace(ccode.group(0),
                                                  ', ' + countries.get(ccode.group(1), ccode.group(1)))
                game = dict()
                gname = bd.find('span.gname>a')
                if gname.len > 0:
                    group = gname.text()[-1]
                    game['name'] = '{0} round, "{1}"'.format(strstage, group)
                    game['descr'] = '{0} round, Group {1}'.format(strstage, group)
                    game['group'] = group
                else:
                    gname = bd.find('span.rname>a')
                    sround = gname.text()
                    game['name'] = '{0} {1}'.format(sleg, sround)
                    game['descr'] = '{0}, {1} leg'.format(sround, sleg)

                tds = bd.find('tr.match_res>td')
                for k in xrange(0, tds.len):
                    classval = tds[k].attr('class')
                    if classval == 'r home nob':
                        game['home_short'] = tds[k].find('a').text()
                    elif classval == 'logo home-logo nob':
                        game['home_full'] = tds[k].find('a>img').attr('title')
                    elif classval == 'l away nob':
                        game['away_short'] = tds[k].find('a').text()
                    elif classval == 'logo away-logo nob':
                        game['away_full'] = tds[k].find('a>img').attr('title')

                summary = '{0} {3} {1} ({2})'\
                            .format(game['home_short'], game['away_short'], game['name'], delim)
                full_summary.append(summary)
                if 'group' in game:
                    groups.add(group)

                event = ical.Event()
                event['uid'] = str(uuid.uuid3(uuid.NAMESPACE_OID, game['home_short'] + game['away_short'] + str(d)))
                event['location'] = stad_name
                event['summary']  = ball_symbol + ' ' + summary
                event['description'] = '{0} vs. {1} ({2}) at {3}'\
                                   .format(game['home_full'], game['away_full'], game['descr'], stad_name)
                event.add('dtstart', dt)
                event.add('dtend', dt + timedelta(hours=2))
                cal.add_component(event)

            if len(full_summary) > 4:
                event = ical.Event()
                event['uid'] = str(uuid.uuid3(uuid.NAMESPACE_OID, str(d)))
                event['location'] = 'Europe'
                #event['summary'] = trophy_symbol + ' {0} {1} round'.format(self.tournament, strstage)
                if not sround and len(groups) > 0:
                    event['summary'] = trophy_symbol + ' {0} {1} round, Group '.format(self.tournament, strstage)\
                                     + ','.join(sorted(groups))
                else:
                    event['summary'] = trophy_symbol + ' {0} {1}'.format(self.tournament, sround)

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

def export_calendar(filename, tournament, byhand=False):
    cal = ical.Calendar()

    exporter = uefa_site_exporter(tournament)
    #exporter.add_group_events(cal)
    exporter.add_playoff_events(cal)

    # No information about matches on site until draw
    if byhand:
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
export_calendar(r'Q:\champions_league.ics', uefa_site_exporter.CHAMPIONS_LEAGUE)
#update_score(r'Q:\el_6_1.ics', 'UEFA Europa League', 6, 1)
#update_score(r'Q:\lc_6_2.ics', 'UEFA Champions League', 6, 2)
