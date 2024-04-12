
from fischer.numbergames.link_m_diff_n import LinkMDiffN



def main():
    game = LinkMDiffN(
        size=(4, 4),
        chain_length=4,
        max_diff=1,
    )

    while True:
        game.randomize_grid(max_n=4)

        print(game)

        for links in game.get_best_partial_solution():
            print(game)
            print(game.get_score(links))



if __name__ == '__main__':
    main()
