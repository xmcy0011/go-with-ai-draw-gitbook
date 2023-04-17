# install gitbook-cli(mac)

1.install nvm
```bash
$ brew install nvm
$ nvm install 10.14.1  # 安装 nodejs 10版本，否则gitbook会出很多错误（很久没更新了）
$ nvm use 10.14.1
$ npm -v
6.4.1
$ node -v
v10.14.1
```

2.install gitbook-cli
```bash
$ npm install gitbook-cli
$ gitbook -v
CLI version: 2.3.2
GitBook version: 3.2.3
```

3. start serve
```bash
$ gitbook init
$ gitbook install
$ gitbook serve
$ gitbook build
```