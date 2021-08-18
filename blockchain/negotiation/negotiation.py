# toy case
from typing import Optional, final

ASKER_STRING: final = "asker"
BIDDER_STRING: final = "bidder"
OPERATOR_STRING: final = "operator"
LOW_ATTRACTOR: final = {
    ASKER_STRING: 0.25,
    BIDDER_STRING: 0.5,
    OPERATOR_STRING: 0.25
}
HIGH_ATTRACTOR: final = {
    ASKER_STRING: 0.25,
    BIDDER_STRING: 0.25,
    OPERATOR_STRING: 0.5
}
CENTRAL_ATTRACTOR: final = {
    ASKER_STRING: 0.1,
    BIDDER_STRING: 0.1,
    OPERATOR_STRING: 0.8
}
LOW_CONCENTRATION: final = {
    ASKER_STRING: 0.25,
    BIDDER_STRING: 0.25,
    OPERATOR_STRING: 0.5
}
HIGH_CONCENTRATION: final = {
    ASKER_STRING: 0.25,
    BIDDER_STRING: 0.25,
    OPERATOR_STRING: 0.5
}
EQUILIBRIUM: final = {
    ASKER_STRING: 0.3,
    BIDDER_STRING: 0.3,
    OPERATOR_STRING: 0.4
}


class NegotiationActor(object):
    def __init__(self, identifier: str = "", balance: float = 0):
        if balance < 0:
            raise ValueError("Balance cannot be negative!")
        self.__identifier = identifier
        self.__balance = balance

    @property
    def identifier(self) -> str:
        return self.__identifier

    @identifier.setter
    def identifier(self, identifier: str):
        raise AttributeError("Identifier attribute cannot be changed!")

    @identifier.deleter
    def identifier(self):
        raise AttributeError("Identifier attribute cannot be deleted!")

    @property
    def balance(self) -> float:
        return self.__balance

    @balance.setter
    def balance(self, balance: float):
        if balance < 0:
            raise ValueError("Balance cannot be negative!")
        self.__balance = balance

    @balance.deleter
    def balance(self):
        raise AttributeError("Balance attribute cannot be deleted!")


class Buyer(NegotiationActor):
    def __init__(self, identifier: str = "", offer: float = 0, balance: float = 0):
        super(Buyer, self).__init__(identifier=identifier, balance=balance)
        if offer < 0:
            raise ValueError("Offer cannot be negative!")
        if offer > balance:
            raise ValueError("Offer cannot be lesser than balance!")
        self.__offer = offer

    @property
    def offer(self) -> float:
        return self.__offer

    @offer.setter
    def offer(self, offer: float):
        if offer < 0:
            raise ValueError("Offer cannot be negative!")
        self.__offer = offer

    @offer.deleter
    def offer(self):
        raise AttributeError("Offer attribute cannot be deleted!")


class Bidder(NegotiationActor):
    def __init__(self, identifier: str = "", proposal: float = 0, balance: float = 0, acceptance: bool = False):
        super(Bidder, self).__init__(identifier=identifier, balance=balance)
        self.__acceptance = acceptance
        self.__proposal = proposal

    @property
    def proposal(self) -> float:
        return self.__proposal

    @proposal.setter
    def proposal(self, proposal: float):
        self.__proposal = proposal

    @proposal.deleter
    def proposal(self):
        raise AttributeError("Proposal attribute cannot be deleted!")

    @property
    def acceptance(self) -> bool:
        return self.__acceptance

    @acceptance.setter
    def acceptance(self, acceptance: bool):
        self.__acceptance = acceptance

    @acceptance.deleter
    def acceptance(self):
        raise AttributeError("Acceptance attribute cannot be deleted!")


class Asker(Buyer):
    def __init__(self, identifier: str = "", offer: float = 0, balance: float = 0, formulating: bool = False):
        super(Asker, self).__init__(identifier=identifier, offer=offer, balance=balance)
        self.__formulating = formulating

    @property
    def formulating(self) -> bool:
        return self.__formulating

    @formulating.setter
    def formulating(self, formulating: bool):
        self.__formulating = formulating

    @formulating.deleter
    def formulating(self):
        raise AttributeError("Formulating attribute cannot be deleted!")

    def generate_offer(self, minimum: float) -> float:
        # TODO: implement more complex algorithm for formulating the offer
        return min(self.balance, minimum)


class NotEnoughOperatorsError(Exception):
    def __init__(self, message="There are not enough additional operators to proceed. At least 3 are required."):
        self.message = message
        super().__init__(self.message)


def transaction(buyer: NegotiationActor, seller: NegotiationActor, amount: float):
    if amount < 0:
        raise ValueError("Amount must non-negative")
    buyer.balance -= amount
    seller.balance += amount


def low_attractor(a: Asker, b: Bidder, *args: Buyer) -> bool:
    low_mean: float = (a.offer + (a.offer + b.proposal) / 2.0) / 2.0
    flag: bool = True

    # se almeno un operatore ha un'offerta superiore ad up_mean allora ritorniamo false
    for operator in args:
        if operator.offer > low_mean:
            flag = False
            break
    return flag


def low_attractor_price(p_bid: float, lm: float, p_ask: float) -> float:
    return LOW_ATTRACTOR[BIDDER_STRING] * p_bid + LOW_ATTRACTOR[OPERATOR_STRING] * lm + LOW_ATTRACTOR[
        ASKER_STRING] * p_ask


def high_attractor(a: Asker, b: Bidder, *args: Buyer) -> bool:
    up_mean: float = (b.proposal + (a.offer + b.proposal) / 2.0) / 2.0
    flag: bool = True

    # se almeno un operatore ha un'offerta inferiore ad up_mean allora ritorniamo false
    for operator in args:
        if operator.offer < up_mean:
            flag = False
            break
    return flag


def high_attractor_price(p_bid: float, um: float, p_ask: float) -> float:
    return HIGH_ATTRACTOR[BIDDER_STRING] * p_bid + HIGH_ATTRACTOR[OPERATOR_STRING] * um + HIGH_ATTRACTOR[
        ASKER_STRING] * p_ask


def central_attractor(a: Asker, b: Bidder, *args: Buyer) -> bool:
    up_mean: float = (b.proposal + (a.offer + b.proposal) / 2.0) / 2.0
    low_mean: float = (a.offer + (a.offer + b.proposal) / 2.0) / 2.0
    flag: bool = True

    # se almeno un operatore ha un'offerta non compresa tra low_mean e up_mean esclusi allora ritorniamo false
    for operator in args:
        if not (low_mean < operator.offer < up_mean):
            flag = False
            break
    return flag


def central_attractor_price(p_bid: float, p_ask: float, p_ops_max: float):
    return CENTRAL_ATTRACTOR[BIDDER_STRING] * p_bid + CENTRAL_ATTRACTOR[OPERATOR_STRING] * p_ops_max + \
           CENTRAL_ATTRACTOR[
               ASKER_STRING] * p_ask


def low_concentration(a: Asker, b: Bidder, *args: Buyer) -> bool:
    up_mean: float = (b.proposal + (a.offer + b.proposal) / 2.0) / 2.0
    flag: bool = False
    count: int = 0

    # contiamo il numero di offerenti con un'offerta <= di up_mean
    for operator in args:
        if operator.offer <= up_mean:
            count += 1

    # se più di 2/3 dei offerenti ha un'offerta <= dell'up_mean allora ritorniamo true
    if count >= int(round(2.0 * len(args) / 3.0)):
        flag = True
    return flag


def low_concentration_price(p_bid: float, p_ask: float):
    p_ops = (2.0 * p_ask + p_bid) / 3.0
    return LOW_CONCENTRATION[BIDDER_STRING] * p_bid + LOW_CONCENTRATION[OPERATOR_STRING] * p_ops + LOW_CONCENTRATION[
        ASKER_STRING] * p_ask


def high_concentration(a: Asker, b: Bidder, *args: Buyer) -> bool:
    low_mean: float = (a.offer + (a.offer + b.proposal) / 2.0) / 2.0
    flag: bool = False
    count: int = 0

    # contiamo il numero di offerenti con un'offerta <= di low_mean
    for operator in args:
        if operator.offer >= low_mean:
            count += 1

    # se più di 2/3 dei offerenti ha un'offerta >= del low_mean allora ritorniamo true
    if count >= int(round(2.0 * len(args) / 3.0)):
        flag = True
    return flag


def high_concentration_price(p_bid: float, p_ask: float):
    p_ops = (p_ask + 2.0 * p_bid) / 3.0
    return HIGH_CONCENTRATION[BIDDER_STRING] * p_bid + HIGH_CONCENTRATION[OPERATOR_STRING] * p_ops + HIGH_CONCENTRATION[
        ASKER_STRING] * p_ask


def equilibrium_price(p_bid: float, p_ask: float, mean: float):
    return EQUILIBRIUM[BIDDER_STRING] * p_bid + EQUILIBRIUM[OPERATOR_STRING] * mean + EQUILIBRIUM[ASKER_STRING] * p_ask


# returns true iff the negotiation ends with the completion of the operation (selling of the goods/service)
def negotiation(a: Asker, b: Bidder, *args: Buyer) -> (bool, Optional[Buyer]):
    winner: Optional[Buyer] = None
    mean: float = (a.offer + b.proposal) / 2.0
    low_mean: float = (a.offer + (a.offer + b.proposal) / 2.0) / 2.0
    up_mean: float = (b.proposal + (a.offer + b.proposal) / 2.0) / 2.0
    success: bool = True  #

    if len(args) < 3:
        raise NotEnoughOperatorsError()

    if low_attractor(a, b, *args):
        negotiation_price = low_attractor_price(b.proposal, low_mean, a.offer)
        if a.formulating and b.acceptance:
            # a vince la negoziazione
            if a.balance >= negotiation_price:
                a.offer = negotiation_price
            winner = a
            transaction(a, b, amount=a.offer)
        elif (not a.formulating) and b.acceptance:
            # troviamo l'operator con l'offerta più vicina a low_mean
            winner = max(*args, key=lambda operator: abs(operator.offer - negotiation_price))
            transaction(winner, b, amount=winner.offer)
        elif not b.acceptance:
            success = False

    elif high_attractor(a, b, *args):
        negotiation_price = high_attractor_price(b.proposal, up_mean, a.offer)
        if a.formulating and b.acceptance:
            # a vince la negoziazione
            if a.balance >= negotiation_price:
                a.offer = negotiation_price
            winner = a
            transaction(a, b, amount=a.offer)
        elif (not a.formulating) and b.acceptance:
            # troviamo l'operator con l'offerta più vicina al prezzo proposto da b
            winner = max(*args, key=lambda operator: abs(operator.offer - negotiation_price))
            transaction(winner, b, amount=winner.offer)
        elif not b.acceptance:
            success = False

    elif central_attractor(a, b, *args):
        # calcoliamo il miglior miglior offerente tra gli operatori
        best_offerer = max(*args, key=lambda operator: abs(operator.offer))
        negotiation_price = central_attractor_price(b.proposal, a.offer, best_offerer)
        if a.formulating and b.acceptance:
            # a vince la negoziazione per il diritto di prelazione facendo un'offerta pari al negotiation_price
            if a.balance >= negotiation_price:
                a.offer = negotiation_price
            winner = a
            transaction(a, b, amount=a.offer)
        elif (not a.formulating) and b.acceptance:
            # troviamo l'operator con l'offerta più vicina a negotiation_price
            winner = max(*args, key=lambda operator: abs(operator.offer - negotiation_price))
            transaction(winner, b, amount=winner.offer)
        elif not b.acceptance:
            success = False

    elif low_concentration(a, b, *args):
        negotiation_price = low_concentration_price(b.proposal, a.offer)
        best_offerer = max(*args, key=lambda operator: abs(operator.offer))
        if a.formulating and b.acceptance:
            # a vince la negoziazione per il diritto di prelazione facendo un'offerta pari a price
            if a.balance >= negotiation_price:
                a.offer = negotiation_price
            winner = a
            transaction(a, b, amount=a.offer)
        elif (not a.formulating) and b.acceptance:
            # troviamo l'operator con l'offerta più alta
            winner = best_offerer
            if winner.balance >= negotiation_price:
                winner.offer = negotiation_price
            transaction(winner, b, amount=negotiation_price)
        elif not b.acceptance:
            success = False

    elif high_concentration(a, b, *args):
        negotiation_price = high_concentration_price(b.proposal, a.offer)
        best_offerer = max(*args, key=lambda operator: abs(operator.offer))
        if a.formulating and b.acceptance:
            # a vince la negoziazione per il diritto di prelazione facendo un'offerta pari a price
            if a.balance >= negotiation_price:
                a.offer = negotiation_price
            winner = a
            transaction(a, b, amount=a.offer)
        elif (not a.formulating) and b.acceptance:
            # troviamo l'operator con l'offerta più alta
            winner = best_offerer
            if winner.balance >= negotiation_price:
                winner.offer = negotiation_price
            transaction(winner, b, amount=negotiation_price)
        elif not b.acceptance:
            success = False

    else:  # situazione di equilibrio
        negotiation_price = equilibrium_price(b.proposal, a.offer, mean)
        best_offerer = max(*args, key=lambda operator: abs(operator.offer))
        if a.formulating and b.acceptance:
            # a vince la negoziazione per il diritto di prelazione facendo un'offerta pari a price
            if a.balance >= negotiation_price:
                a.offer = negotiation_price
            winner = a
            transaction(a, b, amount=a.offer)
        elif (not a.formulating) and b.acceptance:
            # troviamo l'operator con l'offerta più alta
            winner = best_offerer
            transaction(winner, b, amount=negotiation_price)
        elif not b.acceptance:
            success = False

    return success, winner
