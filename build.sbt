lazy val scalaclassifier =
  (project in file(".")).
    settings(
      name := "ScalaClassifier",
      organization := "net.ddns.akgunter",
      version := "0.1",
      scalaVersion := "2.12.6",
      test in assembly := {},
      parallelExecution in Test := false,
      libraryDependencies ++= Seq(
        "com.github.scopt" %% "scopt" % "3.7.0",
        "org.scalaz" %% "scalaz-core" % "7.2.23"
      )
    )
