// =============================================================================
// Jenkins declarative pipeline — Reddit Topics Lakehouse
// -----------------------------------------------------------------------------
// Mirrors the GitHub Actions CI/CD: quality gate (lint/type/test) on every
// build, bundle validate, then gated deploys to staging and prod.
//
// Required Jenkins credentials (Secret text):
//   databricks-host-staging / databricks-token-staging
//   databricks-host-prod    / databricks-token-prod
// Required tools: a Python 3.11 install and the Databricks CLI on PATH.
// =============================================================================
pipeline {
  agent any

  options {
    timestamps()
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '20'))
  }

  environment {
    PIP_DISABLE_PIP_VERSION_CHECK = '1'
    LAKEHOUSE_ENV = 'dev'
  }

  stages {
    stage('Setup') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install -e ".[spark,ingestion,ml,dev]" build
        '''
      }
    }

    stage('Quality') {
      parallel {
        stage('Lint') {
          steps { sh '. .venv/bin/activate && ruff check src tests' }
        }
        stage('Format') {
          steps { sh '. .venv/bin/activate && black --check src tests' }
        }
        stage('Type-check') {
          steps { sh '. .venv/bin/activate && mypy src' }
        }
      }
    }

    stage('Test') {
      steps {
        sh '. .venv/bin/activate && pytest -q --junitxml=reports/junit.xml --cov=reddit_lakehouse'
      }
      post {
        always { junit allowEmptyResults: true, testResults: 'reports/junit.xml' }
      }
    }

    stage('Build wheel') {
      steps {
        sh '. .venv/bin/activate && python -m build --wheel'
        archiveArtifacts artifacts: 'dist/*.whl', fingerprint: true
      }
    }

    stage('Deploy staging') {
      when { branch 'main' }
      steps {
        withCredentials([
          string(credentialsId: 'databricks-host-staging', variable: 'DATABRICKS_HOST'),
          string(credentialsId: 'databricks-token-staging', variable: 'DATABRICKS_TOKEN')
        ]) {
          sh '''
            databricks bundle validate -t staging
            databricks bundle deploy   -t staging
          '''
        }
      }
    }

    stage('Approve prod') {
      when { buildingTag() }
      steps {
        input message: 'Deploy to production?', ok: 'Deploy'
      }
    }

    stage('Deploy prod') {
      when { buildingTag() }
      steps {
        withCredentials([
          string(credentialsId: 'databricks-host-prod', variable: 'DATABRICKS_HOST'),
          string(credentialsId: 'databricks-token-prod', variable: 'DATABRICKS_TOKEN')
        ]) {
          sh '''
            databricks bundle validate -t prod
            databricks bundle deploy   -t prod
          '''
        }
      }
    }
  }

  post {
    failure {
      echo 'Pipeline failed — check the stage logs above.'
    }
    cleanup {
      cleanWs()
    }
  }
}
