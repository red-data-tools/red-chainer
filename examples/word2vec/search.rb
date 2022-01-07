import numpy
import six

n_result = 5  # number of search result to show


open('word2vec.model', 'r') do |f|
  ss = f.readline.split
  (n_vocab, n_units) = ss.map(&:to_i)
  word_index = {}
  index2word = {}
  w = Numo::SFloat.new(n_vocab, n_units)
  f.eachlines.with_index do |line, i|
    ss = line.split
    raise "ss.size(#{ss.size}) != n_units + 1(#{n_units + 1})" unless ss.size == n_units + 1
    word = ss.first
    word_index[word] = i
    w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)
  end
end

s = numpy.sqrt((w * w).sum(1))
w /= s.reshape((s.shape[0], 1))  # normalize

try:
    while True:
        q = six.moves.input('>> ')
        if q not in word2index:
            print('"{0}" is not found'.format(q))
            continue
        v = w[word2index[q]]
        similarity = w.dot(v)
        print('query: {}'.format(q))
        count = 0
        for i in (-similarity).argsort():
            if numpy.isnan(similarity[i]):
                continue
            if index2word[i] == q:
                continue
            print('{0}: {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break

except EOFError:
    pass

