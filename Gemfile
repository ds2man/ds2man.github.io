# frozen_string_literal: true

source "https://rubygems.org"

gemspec

# 20240408 sitemap 위한 plugin 추가
gem "jekyll-sitemap"
gem "jekyll-feed" # [jaoneol]2025-09-17 10:40, CI 에러 해결: jekyll-feed 의존성 추가

gem "html-proofer", "~> 5.2", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
