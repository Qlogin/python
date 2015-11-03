# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:56:12 2015

@author: q-login
"""

def LowerBound(num, n, i):
    return num - num % 10 ** (n - i - 1)

def UpperBound(num, n, i):
    p = 10 ** (n - i - 1)
    return num + p - (num % p)

def PermutationIter(num, snum, n):
    res = -1
    if n < len(snum)-1:
        res = PermutationIter(num, snum, n+1)
    for i in range(n+1, len(snum)):
        snum[n], snum[i] = snum[i], snum[n]
        new_num = int(str(snum))
        if (UpperBound(new_num, len(snum), n) > num) and \
           (LowerBound(new_num, len(snum), n) < res or res == -1):
               if new_num > num and (new_num < res or res == -1):
                   res = new_num
               new_num = PermutationIter(num, snum, n+1)
               if new_num > num and (new_num < res or res == -1):
                   res = new_num
        snum[n], snum[i] = snum[i], snum[n]
    return res

def PermutationStep(num):
    snum = bytearray(str(num))
    return PermutationIter(num, snum, 0)

print PermutationStep(897654321)

###########################################################

def LookSaySequence(num):
    res = []
    for dig in str(num):
        if len(res) == 0 or res[-1] != dig:
            res += [1, dig]
        else:
            res[-2] += 1
    return ''.join(map(str, res))

def CoinDeterminer(num):
    if num < 5:
        return num
    count = 1
    while num - 11 > 0:
        count += 1
        num -= 11
    if num % 2 == 0:
        count += 1
    return count

###########################################################

def GetMinutes(time):
    h,m = map(int, time[:-2].split(':'))
    tag = time[-2:]
    if h == 12 and tag == 'AM':
        h = 0
    elif h != 12 and tag == 'PM':
        h += 12
    return h * 60 + m

def MostFreeTime(strArr):
    GetRanges = lambda x : map(GetMinutes, x.split('-'))
    ranges = map(GetRanges, strArr)
    ranges.sort(cmp = lambda x, y: x[0] < y[0])

    free = 0
    for i in range(1, len(ranges)):
        t = ranges[i][0] - ranges[i-1][1]
        if t > free:
            free = t

    return '%02i:%02i' % (free // 60, free % 60)

MostFreeTime(["12:15PM-02:00PM","09:00AM-12:11PM","02:02PM-04:00PM"])

###########################################################

def CountSeq(num):
   res = {}
   prev = None
   count = 0
   for ch in str(num):
      if ch == prev:
         count += 1
      else:
         res[prev] = max(res.get(prev, 0), count)
         prev = ch
         count = 1
   return res

def TripleDouble(num1,num2):
   count1 = CountSeq(num1)
   count2 = CountSeq(num2)

   for k in count1.iteritems():
      if k[1] >= 3 and count2.get(k[0], 0) >= 2:
         return 1

   return 0

# keep this function call here
# to see how to enter arguments in Python scroll down
#print TripleDouble(465555, 5579)

def PatternChaser(s):
    res_pat = ''
    for i in range(0, len(s)):
        c = s[i]
        j = i + 1
        j = s.find(c, j)
        while j != -1:
            pat = ''
            for k in range(0, min(j - i, len(s) - j)):
                if s[i + k] == s[j + k]:
                    pat += s[i + k]
                else:
                    break
            if len(pat) > len(res_pat):
                res_pat = pat
            j = s.find(c, j+1)
    if len(res_pat) > 1:
        return 'yes ' + res_pat
    else:
        return 'no null'

#print PatternChaser( "sskfssbbb9bbb" )

# Using the Python language, have the function KnightJumps(str) read str which will be a string consisting
# of the location of a knight on a standard 8x8 chess board with no other pieces on the board. The structure
# of str will be the following: "(x y)" which represents the position of the knight with x and y ranging from 1 to 8.
#  Your program should determine the number of spaces the knight can move to from a given location. For example:
# if str is "(4 5)" then your program should output 8 because the knight can move to 8 different spaces from position x=4 and y=5.

def KnightJumps(s):
    x, y = eval(s.replace(' ', ','))
    steps = [(-1, -2), (-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2)]
    board = xrange(1, 9)

    num = 0
    for dx, dy in steps:
        if x + dx in board and y + dy in board:
            num += 1
    return num

def QuickKnight(s):
    x0, y0 = eval(s[:5].replace(' ', ','))
    x1, y1 = eval(s[5:].replace(' ', ','))

    steps = [(-1, -2), (-2, -1), (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2)]
    board_size = xrange(1, 9)
    board = [[0] * 8 for i in board_size]
    board[x0][y0] = 1

    queue = [(x0, y0, 0)]
    while len(queue) > 0:
        x, y, n = queue.pop(0)
        if x == x1 and y == y1:
            return n
        for dx, dy in steps:
            dx += x
            dy += y
            if dx in board_size and dy in board_size and board[dx][dy] == 0:
                board[dx][dy] = 1
                queue.append((dx, dy, n + 1))
    return None


####################################################

import itertools

def isCheck(x1, y1, x2, y2):
    board = range(1, 9)
    if x2 not in board or y2 not in board:
        return None

    if x1 == x2 and y1 == y2:
        return False
    return x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2)

def QueenCheck(strArr):
    x1, y1 = eval(strArr[0].replace(' ', ','))
    x2, y2 = eval(strArr[1].replace(' ', ','))

    if not isCheck(x1, y1, x2, y2):
        return -1

    dirs = [0, 1, -1]
    num = 0
    for dx, dy in itertools.product(dirs, dirs):
        if isCheck(x1, y1, x2 + dx, y2 + dy) == False:
            num += 1
    return num

####
# Using the Python language, have the function PolynomialExpansion(str) take str which will be a string
# representing a polynomial containing only (+/-) integers, a letter, parenthesis, and the symbol "^",
# and return it in expanded form. For example: if str is "(2x^2+4)(6x^3+3)", then the output should be
# "12x^5+24x^3+6x^2+12". Both the input and output should contain no spaces. The input will only contain
# one letter, such as "x", "y", "b", etc. There will only be four parenthesis in the input and your output
# should contain no parenthesis. The output should be returned with the highest exponential element first
#  down to the lowest.
#
# More generally, the form of str will be: ([+/-]{num}[{letter}[{^}[+/-]{num}]]...[[+/-]{num}]...)(copy)
# where "[]" represents optional features, "{}" represents mandatory features, "num" represents integers and
# "letter" represents letters such as "x".
#
# Hard challenges are worth 15 points and you are not timed for them. Use the Parameter Testing feature
# in the box below to test your code with different arguments.

def _add(p, t, v):
    p[t] = p.get(t, 0) + v

def PolyParse(s):
    poly  = {}
    terms = s.replace('-', '+-').replace('^+', '^')
    if terms[0] == '+':
        terms = terms[1:]

    var = set()
    for t in terms.split('+'):
        if t != '':
            sep = t.find('^')
            if sep != -1:
                var.add(t[sep-1:sep])
                power = int(t[sep+1:])
                _add(poly, power, int(t[:sep-1]))
            elif t[-1].isalpha():
                var.add(t[-1])
                _add(poly, 1, int(t[:-1]))
            else:
                _add(poly, 0, int(t))
    assert len(var) < 2
    if len(var) == 0:
        var.add(None)
    return poly, var.pop()

def PolyProd(x, y):
    poly = {}
    for p1, k1 in x.iteritems():
        for p2, k2 in y.iteritems():
            _add(poly, p1 + p2, k1 * k2)
    return poly

def PolyString(poly, var):
    items = list(poly.iteritems())
    items.sort(key=lambda x: x[0], reverse=True)
    def _(x):
        res = str(x[1])
        if x[0] != 0:
            if abs(x[1]) == 1:
                res = res[:-1]
            res += var;
            if x[0] != 1:
                res += '^' + str(x[0])
        return res
    return '+'.join(map(_, items)).replace('+-', '-')

def PolynomialExpansion(s):
    subex = map(PolyParse, s[1:-1].split(')('))
    res = subex[0][0]
    var = set([subex[0][1]])
    for i in xrange(1, len(subex)):
        res = PolyProd(res, subex[i][0])
        var |= set([subex[i][1]])
    var -= set([None])
    return PolyString(res, var.pop() if len(var) > 0 else None)

#print PolynomialExpansion("(1x^2+4)(-1)")

###

def checkRow(board, r, c, v):
    for i in xrange(0, 9):
        if i != c:
            if board[r][i] == v:
                return False
    return True

def checkColumn(board, r, c, v):
    for i in xrange(0, 9):
        if i != r:
            if board[i][c] == v:
                return False
    return True

def checkQuadrant(board, r, c, v):
    count = 0
    for i in xrange(3 * r, 3 * r + 3):
        for j in xrange(3 * c, 3 * c + 3):
            if board[i][j] == v:
                count += 1
    return count == 1


def SudokuQuadrantChecker(strArr):
    board = [eval(i.replace('x', 'None')) for i in strArr]
    res = set()

    for r in xrange(0, 9):
        for c in xrange(0, 9):
            v = board[r][c]
            if v is None:
                continue
            qr, qc = r // 3, c // 3
            qnum = 3 * qr + qc + 1
            if not checkRow(board, r, c, v) or\
               not checkColumn(board, r, c, v) or\
               not checkQuadrant(board, qr, qc, v):
                   res.add(qnum)

    if len(res) == 0:
        return "legal"

    return ','.join([str(i) for i in sorted(res)])

###

def OptimalAssignments(strArr):
    matrix = [eval(i) for i in strArr]
    res  = [None] * len(matrix)
    prev_cost = 0

    for i in xrange(0, len(matrix)):
        cur = i
        new_cost = prev_cost + matrix[i][i]
        res.append(i)

        for mac, task in enumerate(res):
            cost = prev_cost - matrix[mac][task] + matrix[mac][i] + matrix[i][task]
            if cost < new_cost:
                new_cost = cost
                cur = mac
        res[i] = res[cur]
        res[cur] = i
        prev_cost = new_cost

    return ''.join(['({0}-{1})'.format(x+1, y+1) for x, y in enumerate(res)])

#print OptimalAssignments(["(1,1,4)","(5,2,1)","(1,5,2)"])


###
# Using the Python language, have the function NoughtsDeterminer(strArr) take the strArr parameter
# being passed which will be an array of size eleven. The array will take the shape of a Tic-tac-toe
# board with spaces strArr[3] and strArr[7] being the separators ("<>") between the rows, and the rest
# of the spaces will be either "X", "O", or "-" which signifies an empty space. So for example strArr
# may be ["X","O","-","<>","-","O","-","<>","O","X","-"]. This is a Tic-tac-toe board with each row
# separated double arrows ("<>"). Your program should output the space in the array by which any player
# could win by putting down either an "X" or "O". In the array above, the output should be 2 because
# if an "O" is placed in strArr[2] then one of the players wins. Each board will only have one solution
# for a win, not multiple wins. You output should never be 3 or 7 because those are the separator spaces.
###

#import itertools

def checkPosition(board, i, j):
    # Check row
    if board[i][(j+1) % 3] == board[i][(j-1) % 3] != '-':
        return True
    # Check column
    if board[(i+1) % 3][j] == board[(i-1) % 3][j] != '-':
        return True
    # Check main diagonal
    if i == j and board[(i-1) % 3][(j-1) % 3] == board[(i+1) % 3][(j+1) % 3] != '-':
        return True
    # Check secondary diagonal
    if i == 2 - j and board[(i+1) % 3][(j-1) % 3] == board[(i-1) % 3][(j+1) % 3] != '-':
        return True
    return False

def NoughtsDeterminer(strArr):
    board = ''.join(strArr).split("<>")
    for i, j in itertools.product(xrange(3), xrange(3)):
        if board[i][j] == '-' and checkPosition(board, i, j):
            return i * 4 + j
    return None

#print NoughtsDeterminer(["X","O","-","<>","-","O","-","<>","O","X","-"])

# Using the Python language, have the function IntersectingLines(strArr) take strArr which will be
# an array of 4 coordinates in the form: (x,y). Your program should take these points where the first
# 2 form a line and the last 2 form a line, and determine whether the lines intersect, and if they do,
# at what point. For example: if strArr is ["(3,0)","(1,4)","(0,-3)","(2,3)"], then the line created
# by (3,0) and (1,4) and the line created by (0,-3) (2,3) intersect at (9/5,12/5). Your output should
# therefore be the 2 points in fraction form in the following format: (9/5,12/5). If there is no
# denominator for the resulting points, then the output should just be the integers like so: (12,3).
# If any of the resulting points is negative, add the negative sign to the numerator like so: (-491/63,-491/67).
# If there is no intersection, your output should return the string "no intersection". The input points
# and the resulting points can be positive or negative integers.

def fraction(x, y):
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a
    g = gcd(x, y)
    x = x // g
    y = y // g
    if y == 1:
        return str(x)
    return '{0}/{1}'.format(x, y);

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return point(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return point(self.x - other.x, self.y - other.y)
    def __neg__(self):
        return point(-self.x, -self.y)
    def __mul__(self, other):
        if isinstance(other, point):
            return self.x * other.x + self.y * other.y
        return point(self.x * other, self.y * other)
    def normal(self):
        return point(-self.y, self.x)

    @classmethod
    def from_str(cls, s):
        return cls(*eval(s))

class box:
    def __init__(self, *arg):
        self.lo = point(min(arg, key=lambda p: p.x).x, min(arg, key=lambda p: p.y).y)
        self.hi = point(max(arg, key=lambda p: p.x).x, max(arg, key=lambda p: p.y).y)
    def contains(self, p):
        return self.lo.x <= p.x <= self.hi.x and self.lo.y <= p.y <= self.hi.y

def IntersectingLines(strArr):
    no_int = "no intersection"
    l1 = map(point.from_str, strArr[:2])
    l2 = map(point.from_str, strArr[2:])

    box1 = box(*l1)
    box2 = box(*l2)

    if box1.lo.x > box2.hi.x or box1.hi.x < box2.lo.x or box1.lo.y > box2.hi.y or box1.hi.y < box2.lo.y:
        return no_int

    n1 = (l1[1] - l1[0]).normal()
    c1 = n1 * l1[0]
    n2 = (l2[1] - l2[0]).normal()
    c2 = n2 * l2[0]

    det = n1.x * n2.y - n1.y * n2.x
    if det == 0:
        return no_int

    x = c1 * n2.y - n1.y * c2
    y = n1.x * c2 - c1 * n2.x

    if det < 0:
        det = -det
        x = -x
        y = -y

    p = point(x / float(det), y / float(det))
    if not box1.contains(p) or not box2.contains(p):
        return no_int

    return '({0},{1})'.format(fraction(x, det), fraction(y, det))

#print IntersectingLines(["(-3,0)","(-1,4)","(0,-3)","(-2,3)"])
#print IntersectingLines(["(9,-2)","(-2,9)","(3,4)","(10,11)"])
#print IntersectingLines(["(0,15)","(3,-12)","(2,1)","(13,7)"])

def RomanNumeralReduction(s):
    romans = [('I', 1), ('V', 5), ('X', 10), ('L', 50), ('C', 100), ('D', 500), ('M', 1000)]
    rommap = dict(romans)
    val = sum([rommap[ch] for ch in s])
    out = ''
    cur = len(romans) - 1
    while val != 0:
        if romans[cur][1] <= val:
            val -= romans[cur][1]
            out += romans[cur][0]
        else:
            cur -= 1
    return out

#print RomanNumeralReduction("XXXVVIIIIIIIIII")

import math

def SquareFigures(num):
    if num == 1:
        return 0

    value = 10 ** (num - 1)
    return int(round(math.sqrt(value)))

# keep this function call here
# to see how to enter arguments in Python scroll down
#print SquareFigures(6)
#print SquareFigures(7)
#print SquareFigures(8)
#print SquareFigures(10)


def KaprekarsConstant(num):
    count = 0
    while num != 6174:
        snum = '{:04d}'.format(num)
        num = int(''.join(sorted(snum, reverse=True))) - int(''.join(sorted(snum)))
        count += 1
    return count

#print KaprekarsConstant(2111)

def next_sample(idx, i, n):
    idx[i] += 1
    if idx[i] == n and i != 0:
        idx[i] = next_sample(idx, i-1, n-1) + 1
    return idx[i]

def ParallelSums(arr):
    arr.sort()
    s = sum(arr) // 2 # TODO: check that Sum is even
    num = len(arr) // 2

    idx = range(0, num)
    while idx[0] != num + 1:
        if s == sum(map(arr.__getitem__, idx)):
            idx += [x for x in xrange(0, 2 * num) if x not in idx]
            return ','.join(map(lambda x: str(arr[x]), idx))
        next_sample(idx, num - 1, 2 * num)
    return "-1"

print ParallelSums([1, 2, 3, 4, 5, 6])
print ParallelSums([1, 2, 1, 5])

def ArrayCouples(arr):
    out = []
    pairs = [(arr[i], arr[i+1]) for i in xrange(0, len(arr), 2)]
    while len(pairs) != 0:
        x, y = pairs.pop(0)
        if (y, x) in pairs:
            pairs.remove((y, x))
        else:
            out += [x, y]

    return "yes" if len(out) == 0 else ','.join(map(str, out))

#print ArrayCouples([1, 2, 2, 1, 3, 3])

def combination(k, n):
    if n - k < k:
        k = n - k
    a, b = 1, 1
    while k > 0:
        a *= n - k + 1
        b *= k
        k -= 1
    return a // b

def factor(n):
    f = 1
    while n > 1:
        f, n = n * f, n - 1
    return f

def MatchingCouples(arr):
    b, g, n = arr
    n //= 2
    bc = combination(n, b)
    gc = combination(n, g)
    return bc * gc * factor(n)

#print MatchingCouples([5, 10, 4])

def ChessboardTraveling(s):
    (x, y), (a, b) = map(eval, [x.replace(' ', ',') for x in s[1:-1].split(')(')])
    k, n = a - x, b - y
    if n > k:
        n, k = k, n
    cur  = [1] * (n + 1)
    prev = [1] * (n + 1)
    while k > 0:
        for i in xrange(0, n):
            cur[i+1] = cur[i] + prev[i+1]
        cur, prev = prev, cur
        k -= 1
    return prev[n]

#print ChessboardTraveling('(1 1)(5 5)')

###

def SimpleSAT(expr):
    letters = set(filter(str.isalpha, expr)) # get all unique variables
    num = len(letters)
    expr = expr.replace('~', '1-')
    for x in xrange(0, 1 << num):            # bits in 'x' - values of variables
        values = map(int, bin(x)[2:])        # split bits
        values = [0] * (num - len(values)) + values  # add leading zeros
        loc = dict(zip(letters, values))     # 'locals' dictionary
        if eval(expr, None, loc) == 1:
            return "yes"
    return "no"

#print SimpleSAT("(a&b&c)|~a")
#print SimpleSAT("a&(b|c)&~b&~c")

def StepWalking(num):
    # The answer is fibonacci numbers
    x, y = 0, 1
    for i in xrange(0, num):
        x, y = y, x + y
    return y

#print StepWalking(1)
#print StepWalking(3)
#print StepWalking(5)

def ArrayJumping(arr):
    n = len(arr)
    maxe = max(xrange(n), key=lambda i: arr[i])
    queue = [maxe]
    path  = [None] * n
    path[maxe] = 0
    while len(queue) > 0:
        cur = queue.pop(0)
        for nxt in [(cur + arr[cur]) % n, (cur - arr[cur]) % n]:
            if nxt == maxe:
                return path[cur] + 1
            if path[nxt] is None:
                queue.append(nxt)
                path[nxt] = path[cur] + 1
    return -1

#print ArrayJumping([1,2,3,4,2])
#print ArrayJumping([1,7,1,1,1,1])
#print ArrayJumping([1, 2, 3, 2, 1])