return {
  {
    "Yeijon/mnemonic.nvim",
    dependencies = {
      "nvim-telescope/telescope.nvim",
      "stevearc/dressing.nvim",
    },
    config = function()
      local root = vim.fs.root(0, { ".git" }) or vim.fn.getcwd()
      require("mnemonic").setup({
        vault = root .. "/practice",
        daily_limit = 8,
        target_retrievability = 0.90,
        keymaps = {
          new_card = "<leader>la",
          review = "<leader>lr",
          manage = "<leader>lt",
          cards = "<leader>lm",
        },
      })
    end,
  },
}
