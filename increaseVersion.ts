#!/usr/bin/env bun

const fs = require("fs");
const path = require("path");

function incrementVersion(version) {
  let [major, minor, patch] = version.split(".").map(Number);
  patch += 1;
  if (patch >= 999) {
    patch = 0;
    minor += 1;
    if (minor >= 999) {
      minor = 0;
      major += 1;
    }
  }
  return `${major}.${minor}.${patch}`;
}

function updateSetupPy(newVersion) {
  const setupPyPath = path.join(__dirname, "setup.py");
  let setupPyContent = fs.readFileSync(setupPyPath, "utf8");
  setupPyContent = setupPyContent.replace(
    /version=['"](\d+\.\d+\.\d+)['"]/,
    `version='${newVersion}'`
  );
  fs.writeFileSync(setupPyPath, setupPyContent, "utf8");
}

function updatePackageJson(newVersion) {
  const packageJsonPath = path.join(__dirname, "package.json");
  const packageJsonContent = JSON.parse(
    fs.readFileSync(packageJsonPath, "utf8")
  );
  packageJsonContent.version = newVersion;
  fs.writeFileSync(
    packageJsonPath,
    JSON.stringify(packageJsonContent, null, 2),
    "utf8"
  );
}

function main() {
  const setupPyPath = path.join(__dirname, "setup.py");
  const setupPyContent = fs.readFileSync(setupPyPath, "utf8");
  const currentVersionMatch = setupPyContent.match(
    /version=['"](\d+\.\d+\.\d+)['"]/
  );

  if (!currentVersionMatch) {
    console.error("Failed to find version in setup.py");
    process.exit(1);
  }

  const currentVersion = currentVersionMatch[1];
  const newVersion = incrementVersion(currentVersion);

  updateSetupPy(newVersion);
  updatePackageJson(newVersion);

  console.log(`Version updated to ${newVersion}`);
}

main();
