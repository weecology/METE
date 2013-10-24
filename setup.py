try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name = 'mete',
      version= '0.2dev',
      description = 'Tools for analying the Maximum Entropy Theory of Ecology',
      author = "Ethan White, Dan McGlinn, Xiao Xiao, Sarah Supp, and Katherine Thibault",
      url = 'https://github.com/weecology/mete',
      packages = ['mete', 'mete_distributions'],
      package_data = {'mete': ['beta_lookup_table.pck']},
      license = 'MIT',
)
