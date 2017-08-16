module Chainer
  module Utils
    module Array
      def self.force_array(x, dtype=nil)
        # TODO: conversion by dtype
        Numo::NArray.[](*x)
      end
    end
  end
end
