# frozen_string_literal: true

source "https://rubygems.org"

gemspec

# 20240408 sitemap 위한 plugin 추가
gem "jekyll-sitemap"

gem "html-proofer", "~> 5.0", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
