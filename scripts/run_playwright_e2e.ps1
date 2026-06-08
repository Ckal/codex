$ErrorActionPreference = "Stop"

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is not installed. Install Node.js first, then run npm install and npm run e2e:install."
}

npm run e2e
