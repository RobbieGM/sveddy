{ pkgs ? import <nixpkgs> {} }:

let 
  postgres = pkgs.postgresql_16;

  postgresqlSrc = pkgs.fetchurl {
    url = "https://ftp.postgresql.org/pub/source/v${postgres.version}/postgresql-${postgres.version}.tar.bz2";
    sha256 = "sha256-RG6IKU28LJCFq0twYaZG+mBLS+wDUh1epnHC5a2bKVI=";
  };

  pwd = builtins.getEnv "PWD";
  modifiedPostgresql = postgres.overrideAttrs (old: {
    src = postgresqlSrc;

    postInstall = ''
      ${old.postInstall or ""}
      # Copy over the extension files
      mkdir -p $out/share/postgresql/extension
      mkdir -p $lib/lib
      ln -s ${pwd}/sveddy.so $lib/lib
      ln -s ${pwd}/sveddy.control $out/share/postgresql/extension
      ln -s ${pwd}/sveddy--0.1.0.sql $out/share/postgresql/extension
    '';
  });
in
pkgs.mkShell {
  buildInputs = [ modifiedPostgresql ];
  shellHook = ''
    export PATH=${modifiedPostgresql}/bin:$PATH
  '';
}

