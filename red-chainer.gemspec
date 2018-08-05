# coding: utf-8
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "chainer/version"

Gem::Specification.new do |spec|
  spec.name          = "red-chainer"
  spec.version       = Chainer::VERSION
  spec.authors       = ["Yusaku Hatanaka"]
  spec.email         = ["hatappi@hatappi.me"]

  spec.summary, spec.description = "A flexible framework for neural network for Ruby"
  spec.homepage      = "https://github.com/red-data-tools/red-chainer"
  spec.license       = "MIT"
  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_runtime_dependency "numo-narray", ">= 0.9.1.1"
  spec.add_runtime_dependency "red-datasets", ">= 0.0.6"

  spec.add_development_dependency "bundler", "~> 1.15"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "test-unit"
end
